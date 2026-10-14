from typing import List, Dict, Any, Optional, Tuple



from knowledge.process.import_process.base import BaseNode,setup_logging,T
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError, EmbeddingError
from knowledge.process.import_process.config import get_config
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.process.import_process.prompts.item_name_prompt import ITEM_NAME_SYSYTEM_PROMPT,ITEM_NAME_USER_PROMPT

from langchain_core.messages import SystemMessage,HumanMessage
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.milvus_util import get_milvus_client


class ItemNameRecognitionNode(BaseNode):
    name = 'item_name_recognition'

    def process(self,state:ImportGraphState)->ImportGraphState:

        #1、参数校验
        file_title,chunks,config= self._validate_inputs(state)

        #2、构建LLM的上下文
        item_name_context = self._prepare_item_name_context(chunks,config)

        #3、调用模型
        item_name = self._recognition_item_name_by_llm(file_title,item_name_context)

        #4、嵌入商品名(嵌入模型)
        dense_vector,sparse_vector = self._embedding_item_name(item_name)

        #5、存储到Milvus
        self._save_to_milvus(file_title,item_name,dense_vector,sparse_vector,config)

        #6、返回
        self._fill_item_name(self,item_name,state,chunks)
        return state

    def _validate_inputs(self, state:ImportGraphState):
        self.log_step("s1","检验输入参数")

        config = get_config()

        #1、获取state的file title以及chunk
        chunks= state.get("chunks")
        file_title= state.get("file_title")

        #2、判断参数
        if not file_title:
            raise ValidationError("文件标题为空",self.name)

        if not chunks or not isinstance(chunks,list):
            raise ValidationError("chunk无效或者为空",self.name)

        item_name_chunk_k= config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("chunk为空或者无效",self.name)

        self.logger.info(f"检测到文件：{file_title}对应的切片长度{len(chunks)}")
        #3、返回
        return file_title,chunks,config

    def _prepare_item_name_context(self, chunks: Optional[List[Dict[str, Any]]], config):
        self.log_step("s2","商品名提取的上下文")

        result=[]
        #1、判断chunk的类型
        total= 0
        for index,chunk in enumerate(chunks[:config.item_name_chunk_k]):
            if not isinstance(chunk,dict):
                continue
            #2、提取（切片1标题+body）
            content=chunk.get("content")

            spice = f"【切片】-{index+1}-{content}"

            total+= len(spice)
            result.append(spice)

            if total> config.item_name_chunk_size:
                break

        return "\n\n".join(result)[:config.item_name_chunk_size]

            #3、

    def _recognition_item_name_by_llm(self, file_title:str, item_name_context:str):

        #1、实例化LLM客户端（工具）
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error(f'llm调用失败，安全回退商品名{file_title}')
            return file_title

        #2、格式化用户提示词模板
        prompt= ITEM_NAME_USER_PROMPT.format(file_title=file_title,context=item_name_context)


        try:
            llm_response= llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSYTEM_PROMPT),
                HumanMessage(content=prompt)
            ])

            item_name= getattr(llm_response,"content","").strip()

            if not item_name or item_name.upper()=='UNKNOWN':
                self.logger.warning(f"LLM无法提取有效商品名,安全回到标题名{file_title}")
                item_name = file_title
                return item_name
            self.logger.info(f'提取到的商品名{file_title}')
            return item_name
        except Exception as e:
            self.logger.error(f'llm调用失败，安全回退商品名{file_title}')
            return file_title

    def _embedding_item_name(self, item_name:str)->Optional[Tuple[list,dict[Any,Any]]]:
        try:
            #1、获取嵌入模型
            embedding_model= get_bge_m3_embedding_model()
            #2、嵌入item_name
            embedding_result =embedding_model.encode_document([item_name])

            #3、获取稠密、稀疏向量
            dense=embedding_result["dense"][0].tolist()

            start_index =embedding_result["sparse"].indptr[0]
            end_index =embedding_result["sparse"].indptr[1]
            weights = embedding_result["sparse"].data[start_index:end_index].tolist()
            tokenIds = embedding_result["sparse"].indices[start_index:end_index].tolist()
            sparse =dict(zip(tokenIds,weights))
            return dense,sparse

        except Exception as e:
            self.log_step(f"嵌入商品名{item_name}失败，原因是{str(e)}")
            raise EmbeddingError(f"嵌入商品名{item_name}失败，原因是{str(e)}",self.name)

    def _save_to_milvus(
            self,
            state: ImportGraphState,
            file_title: str,
            item_name: str,
            dense_vector: Optional[List[float]],
            sparse_vector: Optional[dict],
            config
    ):
        """保存到 Milvus"""
        self.log_step("step_6", "保存到 Milvus")

        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return

        try:
            # 1. 获取 Milvus 客户端
            client = get_milvus_client()

            # 2. 获取集合名字
            collection_name = "item_name_collection_test"

            # 3. 检查并创建集合
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 4. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name
            }

            # 5. 构建稠密向量
            if dense_vector is not None:
                data["dense_vector"] = dense_vector

            # 6. 构建稀疏向量
            if sparse_vector is not None:
                data["sparse_vector"] = normalize_sparse_vector(sparse_vector)

            # 7. 插入数据
            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已保存到 Milvus，ID: {result['ids'][0]}")

            state["item_name"] = item_name

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")

        # 1. 定义字段
        schema = client.create_schema(enable_dynamic_fields=True)

        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="IP"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")

    def _fill_item_name(self, item_name,str,state:ImportGraphState,chunks:List[Dict[[str,Any]]]):
        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name





