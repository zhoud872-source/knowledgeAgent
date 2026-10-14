import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Tuple, Union,Any,Dict
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode
from knowledge.process.query_process.exceptions import StateFieldError

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_bge_m3_embedding_model



class HyDeSearchNode(BaseNode):
    name = 'hyde_search_node'

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:

        #1、参数校验
        validated_query,validated_item_names = self._validate_query_inputs(state)

        #2、生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validated_item_names)

        #3、获取嵌入模型与milvus客户端
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()
        if not milvus_client or not embedding_model:
            return state

        #4、假设性文档嵌入
        embedding_document= f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents= [embedding_document])
        if not embedding_result:
            return state

        #5、获取item_name的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validated_item_names)

        #6、创建混合请求
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=embedding_result["dense"][0],
            sparse_vector=embedding_result["sparse"][0],
            expr=item_name_filtered_expr,
        )

        #7、执行请求
        reps= execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_requests,
            norm_score=True,
            output_fields=["chunk_id","content","name"]
        )

        if not reps or not reps[0]:
            return state

        return {"hyde_search_result":reps[0]}

    def _validate_query_inputs(self, state: QueryGraphState):
        rewritten_query= state.get('rewritten_query',"")
        item_names = state.get('item_names',"")

        if not rewritten_query or not isinstance(rewritten_query,str):
            raise StateFieldError(node_name=self.name,field_name='rewritten_query',expected_type=str)
        if not item_names or not isinstance(item_names,list):
            raise StateFieldError(node_name=self.name,field_name='item_names',expected_type=list)
        return rewritten_query,item_names

    def _generate_hy_document(self, validated_query, validated_item_names):
        # 获取客户端
        llm_client = get_llm_client()
        if not llm_client:
            return ""

        user_prompt= USER_HYDE_PROMPT_TEMPLATE.format(item_hint=validated_item_names,rewritten_query=validated_query)
        system_prompt= f"你是一位{validated_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            llm_response_content=getattr(llm_response, "content", None).strip()
            if not llm_response_content:
                return ""
            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败：{str(e)}")
            return ""

    def _item_name_filte_expr(self, validated_item_names):
        quoted = ",".join(f"'{v}'" for v in validated_item_names)
        return f"item_name in [{quoted}]"









