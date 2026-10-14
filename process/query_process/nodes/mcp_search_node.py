import asyncio
import json
import logging
from distutils.util import execute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from agents.mcp import MCPServerSse  # pip install openai_agents
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.process.query_process.state import QueryGraphState
from knowledge.process.query_process.base import BaseNode, T
from knowledge.process.query_process.exceptions import StateFieldError


class McpSearchNode(BaseNode):

    name = "mcp_search_node"

    def process(self,state: QueryGraphState)-> Union[QueryGraphState,Dict[str, Any]]:

        #1 参数校验
        validated_rewritten_query,validated_item_names = self._validate_query_inputs(state)

        # 创建mcp client 并让客户端执行工具
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))


        if not mcp_result:
            return state

        # 直接返回state中的部分结果
        return {"web_search_docs":mcp_result}

    def _validate_query_inputs(self, state:QueryGraphState):
        rewritten_query = state.get("rewritten_query","")
        item_names = state.get("item_names","")
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)
        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query):

        # 创建mcp客户端
        mcp_client = MCPServerSse(
            name="通用搜索",
            params={
                "url": self.config.mcp_dashscope_base_url,  # 服务端的端点
                "headers": {"Authorization": self.config.openai_api_key}  # 认证权限 api_key
            },
            cache_tools_list=True,  # mcp服务端工具列表的工具做缓存
        )

        try:
            await mcp_client.connect()
            execute_tool_result = await mcp_client.call_tool(tool_name="bailian_web_search",
                                                             arguments={"query": validated_rewritten_query, "count": 3})

            # 解析工具执行结果
            if not execute_tool_result:
                return []
            if not execute_tool_result.content[0]:
                return []

            text_content_text: str = execute_tool_result.content[0].text
            if not text_content_text:
                return []
            try:
                # 反序列化
                text_content_text: Dict[str, Any] = json.loads(text_content_text)
                pages = text_content_text.get('pages', "")
                if not pages:
                    return []
                search_result = []
                for page in pages:
                    snippet = page.get('snippet', "").strip()
                    title = page.get('title', "").strip()
                    url = page.get('url', "").strip()
                    search_result.append({"snippet": snippet, "title": title, "url": url})
                return search_result
            except Exception as e:
                self.logger.error(f"反序列化失败mcp")
                return []
        finally:
            await  mcp_client.cleanup()  # 关闭连接














