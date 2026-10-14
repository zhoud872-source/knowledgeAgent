import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_bge_m3_embedding_model
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names



class ItemNameAligner():
    """
    查询向量数据库、评分对齐、分数差异过滤
    """
    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3. 分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _match_vector(self, item_names):
        search_results= []

        milvus_client = get_milvus_client()
        if milvus_client is None:
            return []

        # 获取向量模型
        embedding_model = get_bge_m3_embedding_model()
        if embedding_model is None:
            return search_results


        hybrid_embedding_result = generate_hybrid_embeddings(embedding_model, item_names)

        for index,extracted_item in enumerate(item_names):
            # 创建混合检索请求
            hybrid_search_requests= create_hybrid_search_requests(
                dense_vector = hybrid_embedding_result["dense"][index],
                sparse_vector = hybrid_embedding_result["sparse"][index],
            )

            #执行混合检索请求
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name="kb_item_names_v2",
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5,0.5),
                norm_score=True,
                output_fields=("item_name"),
            )

            item_name_search_result = {
                "extracted_item": extracted_item,
                "matches":[
                    {"item_name":h["entity"]["item_name"],"score":h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            search_results.append(item_name_search_result)
        return search_results

    def _item_name_score_align(self, search_results:List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        confirmed = []
        options = []
        for item_name_search_result in search_results:
            extracted_name = item_name_search_result.get("extracted_item")
            matches = sorted(item_name_search_result.get("matches"), key=lambda x: x["score"], reverse=True)
            high = [m for m in matches if m.get("score") > 0.7]

            if high:
                extract = next((h for h in high if str(h["item_name"]) == extracted_name),None)

                if extract:
                    picked = extract.get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)

                elif len(high)==1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)

                else:
                    for h in high[:3]:
                        picked = h["item_name"]
                        if picked not in confirmed and picked not in options:
                            options.append(picked)

            else:
                mid = [m for m in matches if m["score"] >= 0.6 and m.get("item_name") not in confirmed and m.get("item_name") not in options]
                if mid:
                    for m in mid[:3]:
                        picked = m["item_name"]
                        options.append(picked)


        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed:List[str], search_results:List[Dict[str, Any]]) :
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get("matches")
            for m in matches:
                score = m["score"]
                item_name = m["item_name"]
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        sorted_item_name_score= sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1]
        return [name for name, score in item_name_score.items() if max_item_name_score - score <=0.15]


class ItemNameExtractor(BaseNode):
    """
    基于用户原始问题+历史对话，提取用户想问的商品名

    """
    def extract_item_name(self, original_query:str,history_text):
        from knowledge.utils.llm_client_util import get_llm_client
        from knowledge.process.query_process.prompts.item_name_extract_prompt import ITEM_NAME_EXTRACT_TEMPLATE

        result:Dict[str,Any]={"item_names":[],"rewritten_query":""}
        #1 获取llm客户端
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return result

        #2 定义提示词
        human_prompt=ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text if history_text else "暂无上下文",query=original_query)
        system_prompt="你是一个专业的客服助手，擅长理解用户意图和提取关键信息"

        #3 llm调用
        llm_response =llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        llm_content = llm_response.content.strip()

        if not llm_content.strip():
            return result

        #4、清洗和解析
        parsed_result =self._clean_parse(llm_content)
        result['rewritten_query'] =parsed_result.get("rewritten_query") or original_query
        result['item_names']=parsed_result.get("item_names")

    def _clean_parse(self, llm_response:str) -> Dict[str, Any]:

        # 1. 处理 Markdown 代码块围栏（LLM 可能返回 ```json...```）
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        #2、反序列化
        try:
            parsed_llm_result =json.loads(content)
            # 清洗item_names
            raw_item_names =parsed_llm_result.get('item_names')
            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in raw_item_names if raw_item.strip()]

            raw_rewritten_query = parsed_llm_result.get('rewritten_query')

            clean_rewritten_query =""  if not isinstance(raw_rewritten_query,str) else raw_rewritten_query.strip()

            return {"item_names":clean_item_names,"rewritten_query":clean_rewritten_query}


        except JSONDecodeError as e:
            raise JSONDecodeError(f"JSON反序列化失败")



class ItemNameConfirmNode():
    name = 'item_name_confirm_node'

    def __init__(self):
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()



    def process(self, state: QueryGraphState)->QueryGraphState:

        #1、从state获取用户原始问题
        origin_query = state.get("original_query")

        #2 提取商品名llm、对齐下游商品名检索
        clean_llm_result = self._item_name_extractor.extract_item_name(origin_query)

        item_names = clean_llm_result.get("item_names")

        rewritten_query = state.get("rewritten_query")

        if item_names:
            confirmed,options= self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed,options=[],[]



        # 4、决定state的key值
        self._decide(item_names,confirmed,options,rewritten_query)

    def _decide(self, state,item_names:List[str], confirmed:List[str], options:List[str], rewritten_query:str):

        if confirmed:
            state["item_names"]=confirmed
            state["rewritten_query"]=rewritten_query

        elif options:
            state['answer']= f'不确定是哪款产品，是以下产品吗：{"、".join(options)}'

        else:
            state['answer']= "抱歉，没找到"
