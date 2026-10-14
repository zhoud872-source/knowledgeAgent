import json,logging
import re
from dataclasses import field, dataclass
from enum import unique
from json import JSONDecodeError
from typing import List, Dict, Any, Optional, Tuple, Sequence, Set

from tvm_ffi.core import DataType

from knowledge.utils.llm_client_util import get_llm_client
from knowledge.process.import_process.prompts.knowledge_graph_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.process.import_process.base import BaseNode,setup_logging,T
from knowledge.process.import_process.state import ImportGraphState
from knowledge.process.import_process.exceptions import ValidationError, EmbeddingError, MilvusError
from knowledge.process.import_process.config import get_config, ImportConfig
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.neo4j_util import get_neo4j_driver
from langchain_core.messages import HumanMessage,SystemMessage


MAX_ENTITY_NAME_LENGTH = 15


ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}


ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})


CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
"""


CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, item_name: $item_name}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""


CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, item_name: $item_name})
    MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, item_name: $item_name}})
    MATCH (t:Entity {{name: $tail, item_name: $item_name}})
    MERGE (h)-[:{rel_type}]->(t)
"""

CYPHER_CLEAR_ITEM = """
    MATCH (n {item_name: $item_name}) DETACH DELETE n
"""




@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


class Neo4jGraphWriter:
    def __init__(self, database: str = ""):
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)

    def clear(self, neo4j_driver, item_name: str) -> None:
        if not neo4j_driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(neo4j_driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
            self.logger.info(f"Neo4j 旧数据已清理: {item_name}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")



    def insert(self, driver, entities, relations, chunk_id, item_name):
        """neo4j写入"""
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
            self.logger.info(f"Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系")
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")



    def _write_graph_tx(self, tx, entities, relations, chunk_id, item_name):
        # 1. 创建 Chunk 节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. 创建实体节点 + 关联到 Chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description", "")

            # 动态格式化 Cypher，将安全标签注入
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)
            tx.run(cypher_query, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            # 关联实体到 Chunk
            tx.run(CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 创建实体间关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, item_name=item_name)




    def _session(self, driver):

        return driver.session(database=self.database)









class MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus，仅供本模块内部使用。"""

    def __init__(self, collection_name: str):

        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def clear(self, milvus_client:Milvusclient, item_name: str) -> None:
        if not milvus_client:
            raise MilvusError("Milvus 驱动获取失败")

        collection_name = self.collection_name
        try:
            if milvus_client.has_collection(collection_name):
                    milvus_client.delete(
                        collection_name=collection_name,
                        filter= f'item_name=="{item_name}"',
                    )
            self.logger.info(f"Milvus 旧数据已清理: {item_name}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")


    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, content: str, item_name: str) -> None:
        """对外唯一入口：将实体写入 Milvus。"""

        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        # 2. 获取去重后的实体名
        entities_names = list({e["name"] for e in entities})
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        # 3. 获取嵌入模型
        bge_ef_model = get_bge_m3_embedding_model()
        if bge_ef_model is None:
            raise MilvusError("嵌入模型获取失败")

        # 4. 创建集合（不存在则创建）
        try:
            self._ensure_collection(milvus_client, self.collection_name)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}")

        # 5. 嵌入向量化
        try:
            embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}")

        # 6. 构建记录
        records = self._build_records(entities_names, embedded_result, chunk_id, content, item_name)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        # 7. 写入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.debug(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}")

    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx]
            if hasattr(dense, "tolist"):
                dense = dense.tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records




class KnowLedgeGraphNode(BaseNode):
    name = 'knowledge_graph_node'

    def __init__(self, config: Optional[ImportConfig] = None):
        super().__init__(config)
        self.milvus_writer= MilvusEntityWriter(self.config.entity_name_collection)
        self.neo4j_writer= Neo4jGraphWriter(self.config.neo4j_database)





    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 参 数校验

        validated_chunks,item_name= self._validate_get_inputs(state)
        
        stas = ProcessingStats(total_chunks=len(validated_chunks))

        milvus_client = get_milvus_client()
        neo4j_driver = get_neo4j_driver()
        # 删除milvus、neo4j的存在数据
        self._clean_exist_double_data(stas,milvus_client, neo4j_driver,item_name)


        # 批量处理
        self._process_all_chunks(validated_chunks,milvus_client, neo4j_driver,item_name)




    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_item_name

    def _clean_exist_double_data(self, milvus_client, neo4j_driver, item_name):
        """
        删除milvus、neo4j的对应文档记录
        Args:
            milvus_client:
            neo4j_driver:
            item_name:

        Returns:

        """
        #3.1 删除milvus中的item_name=item_name的数据
        """导入前清理该 item_name 下的所有旧数据（Milvus）。"""

        # 1. 清理 Milvus
        self.milvus_writer.clear(milvus_client,item_name)
       
        self.neo4j_writer.clear(neo4j_driver,item_name)





    def _process_all_chunks(self,stas,validated_chunks:List[Dict[str:Any]], milvus_client, neo4j_driver, item_name:str):
        """
        循环处理每一个chunk
        Args:
            validated_chunks:
            milvus_client:
            neo4j_driver:
            item_name:

        Returns:

        """
        for i,chunk in enumerate(validated_chunks):
            if isinstance(chunk, dict):
                continue

            #1.1 获取chunk信息
            chunk_id = chunk.get("chunk_id")
            item_name = chunk.get("item_name")
            content = chunk.get("content")

            try:
                entities_count,relations_count =self._process_single_chunk(chunk_id,content,item_name,milvus_client,neo4j_driver)
                stas.total_entities += entities_count
                stas.total_relations += relations_count
                stas.processed_chunks += 1
                self.logger.info(f"成功处理{chunk_id}/{len(validated_chunks)}")
            except Exception as e:
                stas.failed_chunks += 1
                stas.errors.append(str(e))


    def _process_single_chunk(self, chunk_id:str,content:str, item_name, milvus_client, neo4j_driver):
        #1 调用模型提取实体与关系
        llm_response = self._extract_graph_with_retry(content,item_name)

        #2、清洗jsion围栏 #3、反序列化 #4、清洗实体  #5、清洗关系

        graph_result=self._parse_and_clean(llm_response)
        #2.1 获取解析后的实体、关系
        final_entities= graph_result.get("entities")
        final_relations = graph_result.get("relations")

        #3、写入
        #3.1、将实体名字存储到milvus
        self.milvus_writer.insert(milvus_client,chunk_id,final_entities,content,item_name)


        #3.2、将清洗后的实体以及关系存储到neo4j
        self.neo4j_writer.insert(neo4j_driver,final_entities,final_relations,chunk_id,item_name)



        return len(final_entities), len(final_relations)

    def _extract_graph_with_retry(self, content, item_name):
        #1、获取llm客户端


        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError(f"llm客户端初始化失败")


        for i in range(1,3):
            try:
                #2、调用LLm模型
                response_llm=llm_client.invoke([
                    SystemMessage(content=""),
                    HumanMessage(content=f"切片信息\n\n{content}"),
                ])
                result = getattr(response_llm, 'content', '').strip()
                if result:
                    return result
            except Exception as e:
                if i <3:
                    delay=1*(2**(i-1))
                    self.logger.warning(f'开始第{i}次重试,间隔{delay:1.f}s')
                    time.sleep(delay)
        self.logger.error(f'已经进行了2次重试，失败{str(e)}')

        return

    def _parse_and_clean(self, llm_response)->Dict[str, Any]:
        """
        解析llm返回结果的json代码片段围栏
        反序列化
        获取实体信息 关系
        清洗、返回
        Args:
            llm_response:

        Returns:

        """
        # 判断
        if not llm_response:
            raise ValueError(f'提取chunk信息失败')

        #2、清洗json代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        #3、反序列化
        try:
            parsed_llm_response=json.loads(cleaned)
        except JSONDecodeError as e:
            raise JSONDecodeError(f'失败反序列化，{str(e)}')

        #4、获取实体\关系信息
        entities= parsed_llm_response.get("entities",[])
        relationships = parsed_llm_response.get("relation", [])

        #5、清洗实体
        cleaned_entities = self._clean_entitis(entities)

        #6、获取清洗后的实体名、清洗关系
        cleaned_unique_entity_names = {entity.get('name') for entity in cleaned_entities}

        cleaned_relationships = self._clean_relations(cleaned_unique_entity_names,relationships)

        # 7、构建返回
        return {"entities":cleaned_entities,"relations":cleaned_relationships}



    def _clean_entitis(self, entities):
        """

        Args:
            entities: llm中提取的信息

        Returns:合法的干净的实体信息

        """
        MAX_ENTITY_NAME_LENGTH = 15
        unique_seen=set()
        clean_entities_result = []

        for entity in entities:
            entity_name = str(entity.get("name"),"").strip()

            if not entity_name:
                continue

            if len(entity_name) >= MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:MAX_ENTITY_NAME_LENGTH]

            entity_label = str(entity.get("label"),"").strip()

            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            unique_key=(entity_name,entity_label)

            if unique_key in unique_seen:
                continue
            unique_seen.add(unique_key)

            clean_entities = {"name":entity_name,"label":entity_label}

            entity_description = str(entity.get("description"),"").strip()

            if entity_description:
                clean_entities["description"] = entity_description

            clean_entities_result.append(clean_entities)

        return clean_entities_result

    def _clean_relations(self,cleaned_unique_entity_names:Set[str] ,relationships:List[Dict[str:Any]]) -> List[Dict[str:Any]]:
        """
        清洗关系的头尾节点是否不存在，截取头尾实体名，校验头尾实体名是否有效，校验关系的类型是否在白名单，返回
        Args:
            cleaned_unique_entity_names: 所有唯一的实体名集合
            relationships: LLM输出的关系

        Returns:

        """
        clean_relations_result = []
        #1、遍历所有的关系
        for relation in relationships:
            #1.1、提取头实体名
            head_entity_name = str(relation.get("head"),"").strip()
            #1.2、 提取尾实体名
            tail_entity_name = str(relation.get("tail"),"").strip()

            #1.3、判断头尾实体名是否都在
            if not head_entity_name or not tail_entity_name:
                continue
            #1.4 判断头尾实体名是否过长
            if len(head_entity_name) >= MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) >= MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            #1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entity_names or tail_entity_name not in cleaned_unique_entity_names:
                continue

            #1.6 判断关系的类型
            relation_type = str(relation.get("type"),"").strip()

            #1.7 判断关系类型是否在白名单
            if relation_type not in ALLOWED_RELATION_TYPES:
                relation_type = "default"

            #1.8 构建最终数据结构
            cleaned_relation = {"head":head_entity_name,"tail":tail_entity_name,"type":relation_type}

            clean_relations_result.append(cleaned_relation)

        return clean_relations_result













