from typing import Dict, Any, List, Tuple
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode, setup_logging, T
from knowledge.utils.bge_rerank_util import get_reranker_model





class RerankNode(BaseNode):

    name = 'RerankNode'

    def process(self, state: QueryGraphState):
        # 1. 获取query
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source_docs(state)

        # 3. Rerank精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4. 动态Top_K截取(断崖检测)
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        state['reranked_docs'] = cutoff_docs

        return state

    def _merge_multi_source_docs(self, state: QueryGraphState):
        final_docs = []

        # 1. 获取本地RRF的文档
        for rrf_doc in (state.get('rrf_chunks') or []):
            if not isinstance(rrf_doc, dict):
                continue

            # 1.2 获取文档的内容
            content = rrf_doc.get('content', '').strip()
            if not content:
                continue

            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id')

            # 1.4 格式化本地RRF的chunk结构
            format_rrf_doc = self._format_rrf_docs(content=content, title=title, chunk_id=chunk_id, source="local")
            final_docs.append(format_rrf_doc)



        # 2. 获取web远程的文档
        for web_doc in (state.get('web_search_docs') or []):

            # 2.1 判断当前文档对象类型
            if not isinstance(web_doc, dict):
                continue

            # 2.2 获取文档内容(snippet)
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()

            # 2.3 判断内容
            if not content:
                continue
            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()

            # 2.4 格式化web的文档结构
            format_web_doc = self._format_rrf_docs(content=content, title=title, url=url, source="web")
            final_docs.append(format_web_doc)


        self.logger.info(f"收集到准备进行Rerank精排的文档 {len(final_docs)}")
        return final_docs




    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None, url: str = "", source: str = "") -> Dict[
        str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        # 1. 判断合并后的多源文档是否存在
        if not merged_multi_docs:
            return []

        # 2. 获取reranker模型
        rerank_model = get_reranker_model()
        if rerank_model is None:
            self.logger.error("重排序模型获取失败")
            return []

        # 3. 构建Q->D的pair对 [(Q,D1[content]),(Q,D2:[content]),(Q,Dn)]
        query_doc_content_pairs = [(user_query, doc.get('content')) for doc in merged_multi_docs]
        try:
            # 4. 计算 [-3.558016300201416, 6.255949974060059, -3.1299989223480225, 0.025800857692956924]
            rerank_scores = rerank_model.compute_score(sentence_pairs=query_doc_content_pairs)

            # 5. 映射分数和文档
            score_doc = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

            # 6. 排序 并返回
            sored_score_docs = sorted(score_doc, key=lambda x: x["score"], reverse=True)

            return sored_score_docs
        except Exception as e:
            self.logger.error(f"Rerank重排序失败:{str(e)}")
            return [{**merged_multi_docs, "score": None}]





    def _cliff_cutoff(self, ranked_docs: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
        if not ranked_docs:
            return []

        upper_bound = min(self.config.rerank_max_top_k, len(ranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)
        cutoff_pos = upper_bound

        for i in range(lower_bound - 1, upper_bound - 1):
            current_score = ranked_docs[i].get("score")
            next_score = ranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            abs_gap = current_score - next_score  # 0.3
            rel_gap = abs_gap / (abs(current_score) + 1e-6)

            if abs_gap >= self.config.rerank_gap_abs or rel_gap >= self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                self.logger.debug(
                    f"断崖检测: 位置 {i + 1}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}"
                )
                break

        return ranked_docs[:cutoff_pos]











