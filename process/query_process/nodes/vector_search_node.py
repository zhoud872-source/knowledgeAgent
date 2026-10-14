import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode, T
from knowledge.process.query_process.exceptions import StateFieldError
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model,generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query




class VectorSearchNode(BaseNode):
    name = 'vector_search_node'

    def process(self, state: QueryGraphState)->Union[QueryGraphState,Dict[str, Any]]:

        #1、参数校验
        validated_query,validated_item_names = self._validated_query_inputs(state)

        #2、获取嵌入模型和milvus客户端
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()
        if embedding_model is None or milvus_client is None:
            return state


        #3、对问题进行向量化
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[validated_query])
        if not embedding_result:
            return state

        #4、构建过滤表达式
        item_name_filter_exper = self._item_name_filter(validated_item_names)

        #5、创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=embedding_result['dense'][0],
            sparse_vector=embedding_result['sparse'][0],
            expr=item_name_filter_exper,
            limit=5
        )

        #6、执行请求
        reps= execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name= self.config.chunks_collection,
            search_requests=hybrid_requests,
            norm_score=True,
            output_fields=["chunk_id","content","item_name"]
        )
        if not reps or not reps[0]:
            return state

        return {"embedding_chunks":reps[0]}




    def _validated_query_inputs(self, state):
        #1 获取state的query
        rewritten_query= state.get('rewritten_query','')

        #2 获取item_names
        item_names = state.get('item_names',"")

        if not rewritten_query or not isinstance(rewritten_query,str):
            raise StateFieldError(field_name='rewritten_query',node_name=self.name,expected_type=str)

        if not item_names:
            raise StateFieldError(field_name='item_names',node_name=self.name,expected_type=str)

        return rewritten_query,item_names

    def _item_name_filter(self, validated_item_names: List[str]):
        quoted= ",".join(f'"{v}"' for v in validated_item_names)
        return f"item_name in [{quoted}]"
