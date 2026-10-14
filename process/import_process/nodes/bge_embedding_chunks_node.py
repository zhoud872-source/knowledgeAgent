from typing import List, Dict, Any, Optional, Tuple
from knowledge.process.import_process.base import BaseNode,setup_logging,T
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.process.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
class BgeEmbeddingNode(BaseNode):
    """
    1、获取所有切片
    2、批量嵌入chunks item+content
    3、所有chunks的向量存储到列表中再返回
    """

class BgeEmbeddingChunksNode(BaseNode):
    name = 'bge_embedding_chunks_node'

    def process(self,state:ImportGraphState)->ImportGraphState:
        #1、参数校验
        validated_chunks,config= self._validate_get_inputs(state)

        #2、获取批量嵌入的阈值

        embedding_batch_chunk_size= getattr(config,"embedding_batch_size",16)

        #3、准备分批嵌入
        total_length=len(validated_chunks)
        final_chunks= []
        for i in range(0,total_length,embedding_batch_chunk_size):
            batch= validated_chunks[i:i+embedding_batch_chunk_size]

            final_chunk=self._process_batch_chunks(batch,embedding_batch_chunk_size,total_length)
            final_chunks.extend(final_chunk)

        #4、返回state
        state['chunks']=final_chunks
        return state



    def _validate_get_inputs(self, state:ImportGraphState):
        self.log_step('s1','参数校验')
        config = get_config()
        #1、获取chunks
        chunks = state.get('chunks')

        #2、校验chunks
        if not chunks or not isinstance(chunks,list):
            raise ValidationError(f'chunks is not a list',self.name)

        #3、返回
        return chunks,config

    def _process_batch_chunks(self, batch:List[Dict[str,Any]], embedding_batch_chunk_size:int, total_length:int):
        self.log_step('s2','批量处理文本块的嵌入')
        #1、循环处理所有chunk要嵌入的内容拼接
        embedding_contents= []
        for i,chunk in enumerate(batch):
            content= chunk.get('content')
            item_name= chunk.get('item_name')
            embedding_content= f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        #2、继续循环处理所有chunk的向量，注入
        bge_m3_model = get_bge_m3_embedding_model()
        embedding_result =bge_m3_model.encode_documents(documents= embedding_contents)
        for i,chunk in enumerate(batch):
            # 获取稠密向量
            dense_vector= embedding_result["dense"][i].tolist()
            # 获取稀疏向量
            csr_array= embedding_result["sparse"]
            ind_ptr=csr_array.indptr
            start_ind_ptr=ind_ptr[i]
            end_ind_ptr=ind_ptr[i+1]

            token_id= csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weight= csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            sparse_vector=dict(zip(token_id,weight))

            chunk['dense_vector']= dense_vector
            chunk['sparse_vector']= sparse_vector

        return batch

