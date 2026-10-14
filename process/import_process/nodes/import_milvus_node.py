from dataclasses import field, dataclass
from typing import List, Dict, Any, Optional, Tuple, Sequence


from pymilvus import DataType,MilvusClient

from knowledge.process.import_process.base import BaseNode,setup_logging,T
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.process.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.milvus_util import get_milvus_client


@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype:DataType
    max_length:Optional[int]=None

_SCALAR_FIELDS:Sequence[ScalarFieldSpec]=(
    ScalarFieldSpec(field_name="content",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)
)

class _MilvusSchemaBuilder:
    """
    构建约束
    """
    @staticmethod
    def build(client: MilvusClient,dim:int):
        #1、构建约束对象
        schema =client.create_schema(enable_dynamic_field=True)
        #2、构建主键
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True
        )
        #3、构建向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        # 构建标量字段
        for scalar_field in _SCALAR_FIELDS:
            kwargs:Dict[str, Any]={"field_name": scalar_field.field_name,"datatype": scalar_field.datatype}
            if scalar_field.max_length:
                kwargs["max_length"] = scalar_field.max_length

            schema.add_field(**kwargs)
        return schema


class _MilvusIndexBuilder:
    """
    创建索引
    """
    @staticmethod
    def build(client: MilvusClient,collection_name:str):
        index= client.prepare_index_params(collection_name=collection_name)

        index.add_index(
            field_name= "dense_vector",
            index_name= "dense_vector_name",
            index_type= "AUTOINDEX",
            metric_type= "COSINE",
        )

        index.add_index(
            field_name= "sparse_vector",
            index_name= "sparse_vector_name",
            index_type= "SPARSE_INVERTED_INDEX",
            metric_type= "IP",
        )


class _MilvusInserter:
    """
    将数据插入到milvus，以及回填
    """
    def __init__(self,client:MilvusClient,collection_name:str):
        self._client = client
        self._collection_name = collection_name

    def insert(self,chunks:List[Dict[str,Any]])->List[Dict[str,Any]]:
        inserted_result= self._client.insert(collection_name=self._collection_name,chunks=chunks)

        inserted_count= inserted_result.get("insert_count")

        ids = inserted_result.get("ids")

        self._fill_chunk_ids(chunks,ids)

    def _fill_chunk_ids(self, chunks:List[Dict[str,Any]], ids:List[Any]):
        for chunk,id in zip(chunks,ids):
            chunk["chunk_id"] = id



class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"
    def process(self,state: ImportGraphState)->ImportGraphState:
        #1、参数校验
        validated_chunks,dim,config =self._validate_get_inputs(state)

        #2、获取milvus客户端
        milvus_client =get_milvus_client()

        if milvus_client is None:
            return state

        collection = getattr(config, "chunks_collection","test_collection")
        #4、确保集合存在
        self._ensure_collection(milvus_client,collection,dim)

        #5、插入
        inserter = _MilvusInserter(milvus_client,collection)

        final_chunks =inserter.insert(validated_chunks)

        state["chunks"] = final_chunks

        return state




    def _validate_get_inputs(self, state):
        self.log_step("s1","参数校验")
        config = get_config()

        chunks =state.get("chunks")

        if not chunks:
            raise ValidationError("待导入的chunk不存在",self.name)

        validated_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error('待入库的chunk不存在')

        # 判断有效集合
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效",self.name)

        # 获取向量维度
        dim = len(validated_chunks[0].get("dense_vector"))
        return validated_chunks,dim ,config

    def _ensure_collection(self, milvus_client:MilvusClient, collection_name:str,dim:int,delete_flag:bool=True):

        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)
            self.logger.info(f"集合{collection_name}被删除")

        #1、判断集合
        if milvus_client.has_collection(collection_name=collection_name):
            self.logger(f"{collection_name}已经存在")
            return

        #创建约束
        schema =_MilvusSchemaBuilder.build(milvus_client,dim)

        #创建索引
        index = _MilvusIndexBuilder.build(milvus_client,collection_name)

        #创建集合
        milvus_client.create_collection(collection_name=collection_name,schema=schema,index=index)
