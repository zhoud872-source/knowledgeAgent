import logging, re, json

from knowledge.process.import_process.nodes.kg_graph_node import MAX_ENTITY_NAME_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple, Union
from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode, T
from knowledge.process.query_process.exceptions import StateFieldError
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query, \
    fetch_chunks_by_chunk_ids
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.neo4j_util import get_neo4j_driver



_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN = 0.5


#-------------------------------------------------
# Neo4J的信息
# -------------------------------------------------
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]

# Neo4j的Cypher语句
_CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN  n.item_name as item_name,n.name as name
LIMIT 1
"""

# toLower()小写
_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($name)
      AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""

# 查询种子节点的一跳关系
_CYPHER_ONE_HOP_RELATIONS = """

MATCH (seed:Entity {name:$name,item_name:$item_name})-[r]-(nbr:Entity)

WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name

RETURN 
  CASE WHEN startNode(r)=seed  THEN  seed.name  ELSE nbr.name END AS head,
  type(r) as rel,
  CASE WHEN  startNode(r)=seed  THEN nbr.name ELSE seed.name END AS tail

limit $limit
"""
# 根据带权重的节点查询chunk_id
_CYPHER_LOOKUP_CHUNK = """

UNWIND $weighted_nodes as n

MATCH (e:Entity{name:n.entity_name,item_name:n.item_name})-[r:MENTIONED_IN]->(c:Chunk{item_name:n.item_name})

WITH c,sum(n.weight) AS score, count(e) AS cnt

RETURN c.id AS chunk_id, c.item_name AS item_name, score, cnt

ORDER BY score DESC, cnt DESC,chunk_id DESC

LIMIT $limit

"""

SEED_NODE_WEIGHT = 2.0
NER_NODE_WEIGHT = 1.0





def _clean_parse_llm_content(llm_response_content):
    if not llm_response_content:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", llm_response_content)
    re_sub = re.sub(r"\s*```$", "", text)

    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except JSONDecodeError as e:
        logging.error(f"JSON 反序列失败，原因: {str(e)}")
        return []
    entities_name = deserialized_result.get('entities', [])
    if not entities_name:
        return []
    if not isinstance(entities_name, list):
        return []

    seen = set()
    entity_name_result = []
    for entity_name in entities_name:
        if not entity_name:
            continue
        if not isinstance(entity_name, str):
            continue

        truncate_entity_name = truncate_entity_name_length(entity_name)

        if truncate_entity_name not in seen:
            seen.add(truncate_entity_name)
            entity_name_result.append(truncate_entity_name)

    return entity_name_result





def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name







class _EntityExtractor:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self,user_query:str):
        llm_client = get_llm_client(response_format=True)

        if not llm_client:
            return []

        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(MAX_ENTITY_NAME_LENGTH=MAX_ENTITY_NAME_LENGTH)

        try:
            llm_response= llm_client.invoke(
                SystemMessage =entities_name_extract_system_prompt,
                HumanMessage = f"用户问题：{user_query}"
            )

            llm_response_content = getattr(llm_response, 'content',"").strip()

            entities_name = _clean_parse_llm_content(llm_response_content)
            return entities_name
        except Exception as e:
            self._logger.error(f"llm实体提取失败：{str(e)}")
            return []


def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"



class _EntityAligner:
    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name


    def align(self,entity_names:List[str],item_names:List[str]):

        fallback_result = {"align_entity_names":[],"align_entity_elements":[]}

        if not entity_names:
            return fallback_result

        embedding_model = get_bge_m3_embedding_model()
        if embedding_model is None:
            self._logger.error('嵌入模型不在')
            return fallback_result

        milvus_client = get_milvus_client()
        if not milvus_client:
            self._logger.error('milvus获取失败')
            return fallback_result

        embedding_result = generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=entity_names)

        if embedding_result is None:
            self._logger.error('实体嵌入失败')
            return fallback_result

        embedding_result_dense = embedding_result['dense']
        embedding_result_sparse = embedding_result['sparse']
        item_name_filtered_expr = _item_name_filter_expr(item_names)

        align_entity_names = []
        align_entity_elements = []
        seen= set()

        for index,entity_name in enumerate(entity_names):

            align_one_result:List[Dict[str, Any]] = self._align_one(milvus_client,
                                                                    self._collection_name,
                                                                    item_name_filtered_expr,
                                                                    embedding_result_dense,
                                                                    embedding_result_sparse,
                                                                    index,
                                                                    entity_name)
            align_entity_elements.extend(align_one_result)

            for detail in align_one_result:
                aligned_name = detail.get('aligned')
                item_name = detail.get('item_name')

                if aligned_name:
                    key = (item_name, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        align_entity_names.append(aligned_name)

        self._logger.info(f"对齐后的实体个数 {len(align_entity_names)} 实体的名字：{align_entity_names}")
        return {
            "entities_aligned_name": align_entity_names,
            "entities_aligned_elements": align_entity_elements
        }



    def _align_one(self, milvus_client, _collection_name, item_name_filtered_expr, embedding_result_dense,
                   embedding_result_sparse, index, entity_name):

        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]


        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}]

        hybrid_search_requests = create_hybrid_search_requests(dense_vector=dense_vector,
                                                              sparse_vector=sparse_vector,
                                                              expr= item_name_filtered_expr,
                                                              limit=5
                                                              )
        reps= execute_hybrid_search_query(milvus_client=milvus_client,
                                          collection_name=_collection_name,
                                          search_requests=hybrid_search_requests,
                                          ranker_weights=(0.5,0.5),
                                          norm_score=True,
                                          limit=5,
                                          output_fields=["source_chunk_id", "item_name","context","entity_name"],
                                          )
        hits= reps[0] if reps else None
        if not hits:
            return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]

        best_by_item: Dict[str, Dict] = {}

        for hit in hits:
            entity_name = hit["entity_name"]
            item_name = hit["item_name"]

            if item_name not in best_by_item:
                best_by_item[item_name] = hit

            if not best_by_item:
                return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]


        results=[]
        for item_name,best in best_by_item.items():
            score = best["distance"]
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN ):
                continue
            ent =best.get('entity_name')

            results.append({
                    "original": entity_name,
                    "aligned": ent.get("entity_name"),
                    "score": score,
                    "item_name": item_name,
                    "source_chunk_id": ent.get("source_chunk_id"),
                    "reason": "top1_per_item",
                })

        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results


def _clean_seed_rows(rows: List[Dict[str, Any]]):
    if not rows:
        return []

    clean_seeds_result: List[EntitySeedNode] = []
    # 1. 遍历
    for row in rows:
        # 1.1 获取item_name
        item_name = row.get('item_name', '').strip()
        # 1.2 获取entity_name
        entity_name = row.get('name', '').strip()
        # 1.3 判断
        if not item_name or not entity_name:
            continue
        # 1.4 封装一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })
    # 2. 返回
    return clean_seeds_result


class _Neo4jGraphReader:
    def __init__(self,
                 database: str,
                 kg_max_seed_candidates: int,
                 kg_max_total_seeds: int,
                 kg_max_triples_per_seed: int,
                 kg_max_total_triples: int,
                 kg_max_total_chunks: int
                 ):
        self._database = database
        self._kg_max_seed_candidates = kg_max_seed_candidates
        self._kg_max_total_seeds = kg_max_total_seeds
        self.kg_max_triples_per_seed = kg_max_triples_per_seed
        self._kg_max_total_triples = kg_max_total_triples
        self._kg_max_total_chunks = kg_max_total_chunks
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        # 1.获取驱动
        neo4j_driver = get_neo4j_driver()

        # 2. 判断驱动是否存在
        if neo4j_driver is None:
            raise RuntimeError("Neo4J驱动获取失败")

        # 3. 返回session对象
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self,pairs:List[ItemEntityPair]):
        if not pairs:
            return []

        final_seed_result = []
        for pair in pairs:
            item_name = pair.get('item_name',"").strip()
            entity_name = pair.get('entity_name',"").strip()
            if not item_name or not entity_name:
                continue

            try:
                with self._session() as session:
                    candidates_seed_nodes = self._excute_seed_nodes(session, item_name,entity_name,self._kg_max_seed_candidates)

                    final_seed_result.extend(candidates_seed_nodes)

                    if len(final_seed_result) > self._kg_max_total_seeds:
                        final_seeds_result = final_seed_result[:self._kg_max_total_seeds]
                        break
            except Exception as e:
                self._logger.error(f"种子节点获取失败：{str(e)}")

        self._logger.info(f"获取种子节点个数{len(final_seed_result)}")
        return final_seed_result

    def _excute_seed_nodes(self, session, item_name, entity_name, _kg_max_seed_candidates):
        exact_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_EXACT_SEEDS, item_name=item_name, name=entity_name
            ).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)

        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_FUZZY_SEEDS, item_name=item_name, name=entity_name, limit=_kg_max_seed_candidates
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)






    def _execute_one_hop_relations(self, session, item_name, seed_name, kg_max_triples_per_seed):
        one_hop_relations = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_ONE_HOP_RELATIONS, item_name=item_name, name=seed_name, limit=kg_max_triples_per_seed
            ).data()
        )

        # 2. 解析结构
        if not one_hop_relations:
            return []

        # 3. 遍历所有的一条关系
        one_hop_relations_result = []
        for one_hop_relation in one_hop_relations:
            # 3.1 提取head
            head = one_hop_relation.get('head', '').strip()
            # 3.2 提取rel
            rel = one_hop_relation.get('rel', '').strip()
            # 3.3 提取tail
            tail = one_hop_relation.get('tail', '').strip()

            # 3.4 判断是否存在关系链
            if not (head and rel and tail):
                continue

            # 3.5 将关系链添加到最终结果中
            one_hop_relations_result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "item_name": item_name
            })
        return one_hop_relations_result




    def find_one_hop_relations(self, seed_nodes):
        #1、判断种子节点
        if not seed_nodes:
            return []
        seen = set()
        one_hop_relations_final_result = []

        #2、遍历所有种子节点
        for seed_node in seed_nodes:
            # 2.1 提取item_name
            item_name = seed_node.get('item_name', "")
            # 2.2 提取entity_name
            seed_name = seed_node.get('entity_name', "")
            # 2.3 判断是否都存在
            if not item_name or not seed_name:
                continue

            # 执行cypher语句


            with self._session() as session:

                seed_one_hop_relations: List[OneHopRelation] = self._execute_one_hop_relations(session, item_name,
                                                                                               seed_name,
                                                                                               self.kg_max_triples_per_seed)

    def collect_node_weight(self, seed_nodes, one_hop_relations):
        """
          职责：
          为种子节点设置权重weight:高=2.0
          为邻居节点设置权重weigh:低=1.0
         Args:
             seed_nodes: 种子节点
             one_hop_relations:一跳关系

         Returns:
          所有节点带权重
         """

        # 1. 判断seed_nodes 是否存在
        if not seed_nodes:
            return []

        # 2. 判断 one_hop_relations 是否存在
        if not one_hop_relations:
            return []
        # 存放所有节点（种子节点和邻居节点的权重）
        weight_map: [Tuple[str, str], float] = {}
        seen = set()
        # 3. 遍历所有的种子节点
        for seed_node in seed_nodes:
            # 3.1 获取item_name
            item_name = seed_node.get('item_name')
            # 3.2 获取节点名
            seed_name = seed_node.get('entity_name')

            key = (item_name, seed_name)
            if key not in seen:
                seen.add(key)
                weight_map[key] = SEED_NODE_WEIGHT

        # 4. 遍历一跳三元组
        for one_hop_relation in one_hop_relations:

            # 4.1 获取head
            head = one_hop_relation.get('head')
            # 4.2 获取tail
            tail = one_hop_relation.get('tail')

            # 4.3 获取商品的名字
            item_name = one_hop_relation.get('item_name')

            # 4.4 为邻居节点赋值权重
            if head and (item_name, head) not in weight_map:
                weight_map[(item_name, head)] = NER_NODE_WEIGHT

            if tail and (item_name, tail) not in weight_map:
                weight_map[(item_name, tail)] = NER_NODE_WEIGHT

        return [{"item_name": it, "entity_name": en, "weight": w}
                for (it, en), w in weight_map.items()]

    def find_nodes_chunk_id(self, weighted_nodes):
        # 1. 执行Cypher语句
        try:
            with self._session() as session:
                # 1.1 查询语句
                sorted_node_chunk_id = session.execute_read(lambda tx: tx.run(
                    _CYPHER_LOOKUP_CHUNK, weighted_nodes=weighted_nodes, limit=self._kg_max_total_chunks
                ).data())


        except Exception as e:
            self._logger.error(f"反查chunk_id失败,原因:{str(e)}")
            return []

        hits = []
        # 2. 处理结果
        for chunk_row in sorted_node_chunk_id:
            chunk_id = chunk_row.get('chunk_id', "").strip()
            item_name = chunk_row.get('item_name', "").strip()
            score = chunk_row.get('score')

            if chunk_id and item_name:
                hits.append({
                    "id": None,
                    "distance": float(score or 0.0),
                    "entity": {"chunk_id": str(chunk_id), "item_name": str(item_name)}
                })

        return hits


class _ChunkBackFiller:
    """

    职责：
    1、根据Neo4J返回的chunk信息（entity{"chunk_id"}） 获取到chunk_ids
    2、根据chunk_ids 查询milvus 获取到chunks(批量操作，返回的chunk没有所谓的顺序，只负责将chunk_id的对象给你)：导致根据分数降序的chunk_id就失去了作用
    3、构建一个chunk_id和chunk对象的字典映射表，将milvus返回的chunk_id和chunk对象存储进来
    4、遍历原有分数降序的chunk_id列表，然后在从该映射表中获取对应的chunk(保证原有根据分数降序的chunk_id)顺序能利用起来

    3、更新到state

    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def back_fill(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        职责：
        1. 获取所有chunk_id
        2. 根据批量的chunk_id查询milvus
        3. 构建chunk_id和chunk对象的映射表（内存操作 快且没有oom风险（batch））
        4. 遍历原有顺序的chunk_id列表(有分数顺序)，接着去到映射表中获取对应的chunk（有分数顺序）
        Args:
            chunk_nodes_sorted: 已经根据排序规则排好顺序的chunk节点

        Returns:

        """

        # 1. 判断chunk_nodes_sorted是否存在
        if not chunk_nodes_sorted:
            return []

        # 2. 获取chunk_ids
        chunk_ids: List[Union[str, int]] = self._collect_chunk_ids(chunk_nodes_sorted)

        # 3. 查询milvus 获取chunk对象
        try:
            chunks: List[Dict[str, Any]] = fetch_chunks_by_chunk_ids(
                collection_name=self._collection_name,
                chunk_ids=chunk_ids,
                output_fields=['chunk_id', 'content', 'title', 'item_name'],
                batch_size=30
            )
            if not chunks:
                return []
        except Exception as e:
            self.logger.error(f"根据chunk_id批量查询chunk对象失败：{str(e)}")
            return []

        # 4. 构建chunk_id和chunk对象的映射表
        chunk_id_map = {str(chunk.get('chunk_id')): chunk for chunk in chunks if chunk.get('chunk_id') is not None}

        # 5. 根据真实顺序的chunk_id从映射表查询真正的chunk对象
        return [{"entity":chunk_id_map.get(str(chunk_id))} for chunk_id in chunk_ids]

    def _collect_chunk_ids(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Union[str, int]]:
        """

        Args:
            chunk_nodes_sorted:

        Returns:

        """
        chunk_ids = []
        # 1. 遍历chunk_ids
        for chunk_node in chunk_nodes_sorted:

            # 2. 判断chunk_id
            if not chunk_node:
                continue

            # 3. 获取entity
            entity = chunk_node.get('entity', '')
            if not entity:
                continue

            # 4. 获取chunk_id
            chunk_id = entity.get('chunk_id')
            if not chunk_id:
                continue

            # 5. chunk_id转换
            chunk_id_str = str(chunk_id)
            try:
                chunk_id_int = int(chunk_id_str)
                chunk_ids.append(chunk_id_int)
            except (ValueError, TypeError):
                # 出现转换失败的异常很小
                chunk_ids.append(chunk_id_str)

        return chunk_ids


def _build_item_entity_pairs(align_entities_info):
    if not align_entities_info:
        return []

    seen = set()
    item_entity_pairs = []

    for align_entity_info in align_entities_info:
        item_name = align_entity_info.get("item_name").strip()
        align_entity_name = align_entity_info.get("aligned").strip()

        if not (item_name and align_entity_name):
            continue

        key = (item_name, align_entity_name)


        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": align_entity_name
            })
        # 3. 返回
    return item_entity_pairs


def _one_hop_relations_to_texts(triples):
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        it = (tr.get("item_name") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}")
    return docs



class KnowledgeGraphSearchNode(BaseNode):
    """
         知识图谱查询主编排器。

         职责：
         - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
         - 按 pipeline 顺序编排调用

         Pipeline:
         ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
            抽取实体  ──▶   对齐实体   ──▶    Neo4j查询   ──▶ 回填chunk
         └──────────┘   └──────────┘   └────────────┘   └──────────┘
         """
    name = "kg_search_node"
    def process(self, state: QueryGraphState):
        validated_query,validated_item_names= self._validate_inputs(state)

        # 执行流水线
        kg_result = self._run_pipeline(validated_query, validated_item_names)

        return {
            "kg_chunks": kg_result.get("kg_chunks"),
            "kg_triples": kg_result.get("kg_result")

        }







    def _validate_inputs(self, state):
        rewritten_query= state.get("rewritten_query","")
        item_names = state.get("item_names","")

        if rewritten_query is None or not isinstance(rewritten_query,str):
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)
        if item_names is None or not isinstance(item_names,str):
            raise StateFieldError(node_name=self.name,field_name="item_names",expected_type=str)

        user_query = rewritten_query
        for name in item_names:
            if not name:
                continue
            pattern = r"\s*".join(re.escape(ch) for ch in name.replace(" ", ""))
            user_query = re.sub(pattern, "", user_query, flags=re.IGNORECASE)

        user_query = " ".join(user_query.split()).strip()

        return user_query,item_names


    def _run_pipeline(self, validated_query:str, validated_item_names:List[str]):

        #1、初始化组件
        entity_extractor= _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name= self.config.entity_name_collection)
        neo4j_graph_reader = _Neo4jGraphReader(database=self.config.neo4j_database,
                                               kg_max_seed_candidates= self.config.kg_max_seed_candidates,
                                               kg_max_total_seeds=self.config.kg_max_total_seeds,
                                               kg_max_triples_peer_seed= self.config.kg_max_triples_peer_seed,
                                               kg_max_total_triples=self.config.kg_max_total_triples,
                                               kg_max_total_chunks=self.config.kg_max_total_chunks)

        chunk_back_filler= _ChunkBackFiller(collection_name= self.config.chunks_collection)

        #2、各个组件执行各种任务
        #2.1 利用提取器组件、对齐组件提取实体及对齐后的实体
        entity_names = entity_extractor.extract(validated_query)
        entity_align_names= entity_aligner.align(entity_names,item_names=validated_item_names)

        # 获取所有对齐后的实体名
        entity_aligned_names=entity_align_names.get('entities_aligned_name')
        align_entities_info = entity_align_names.get('entities_aligned_elements')
        item_entity_pairs = _build_item_entity_pairs(align_entities_info)

        seed_nodes= neo4j_graph_reader.find_seed_nodes(item_entity_pairs)

        one_hop_relations: List[OneHopRelation] = neo4j_graph_reader.find_one_hop_relations(seed_nodes)

        # 根据种子节点及一跳关系分别设置权重
        weighted_nodes: List[Dict[str, Any]] = neo4j_graph_reader.collect_node_weight(seed_nodes, one_hop_relations)

        # 根据带权重的节点反查chunks
        chunk_nodes_sorted: List[Dict[str, Any]] = neo4j_graph_reader.find_nodes_chunk_id(weighted_nodes)

        # 查找milvus回填chunk
        kg_chunks = chunk_back_filler.back_fill(chunk_nodes_sorted)

        # 3. 将一跳关系转换成模型能够理解的真实图谱结构
        triples_docs = _one_hop_relations_to_texts(one_hop_relations)

        # 4. 汇总知识图谱节点的所有信息
        return {
            "kg_chunks": kg_chunks,  # 回填后的切片文本 → 送入 RRF
            "kg_triples": triples_docs,  # 关系文本描述 → 送入答案生成 prompt
            "kg_seed_nodes": seed_nodes,
            "kg_triples_raw": one_hop_relations,
            "kg_entities": entity_names,
            "kg_aligned_entities": entity_align_names,
            "kg_alignments": align_entities_info,
        }








