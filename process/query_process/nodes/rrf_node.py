from typing import Dict, Any, List, Tuple
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode, setup_logging



class RrfNode(BaseNode):
    name = 'rrf_node'

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get('embedding_chunks') or []

        # 1.2 获取hyde向量检索路的结果
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # 1.3 获取kg图谱检索的结果
        kg_search_chunks = state.get('kg_chunks') or []

        # 2. 为不同路的搜索结果设置不同的权重
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result": (self._normalize_input(hyde_search_chunks), 1.0),
            "kg_search_result": (self._normalize_input(kg_search_chunks), 0.7)
        }

        # 3. 构建rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(rrf_inputs, self._rrf_k, self._top_k)

        # 5. 获取rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # 6. 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回state
        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        diff_path_result = []

        if not rrf_input:
            return []

        for doc in rrf_input:

            # 2.1 判断是否有效
            if not isinstance(doc, dict):
                continue

            # 2.2  获取entity
            entity = doc.get('entity')
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, _rrf_k, _top_k):

        chunk_scores = {}  # 存放所有chunk（多路命中的chunk只留一份）的rrf计算后后的分数值
        chunk_data = {}  # 存放所有chunk

        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, float(0)) + weight / (_rrf_k + i)
                chunk_data.setdefault(chunk_id, doc)

        sorted_results = sorted([(chunk_data[cid], score) for cid, score in chunk_scores.items()],
                                key=lambda x: x[1], reverse=True)

        return sorted_results[:_top_k] if _top_k else sorted_results





