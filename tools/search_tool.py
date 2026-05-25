"""SerpAPI 搜索工具 —— 为 Agent 提供互联网搜索能力"""
import json
import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SearchToolInput(BaseModel):
    """搜索工具输入参数"""
    query: str = Field(..., description="搜索查询字符串")


class SearchTool(BaseTool):
    """使用 SerpAPI 搜索互联网，获取竞品分析相关信息"""
    name: str = "search_internet"
    description: str = (
        "使用SerpAPI搜索互联网，获取与竞品分析相关的最新文章、新闻和信息。"
        "输入搜索查询字符串，返回前5条搜索结果的标题、链接和摘要。"
    )
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return json.dumps(
                {"error": "未配置 SERPAPI_API_KEY，请在环境变量或侧边栏中设置"},
                ensure_ascii=False,
            )

        try:
            from serpapi import GoogleSearch
        except ImportError:
            return json.dumps(
                {"error": "serpapi 包未安装，请执行: pip install google-search-results"},
                ensure_ascii=False,
            )

        try:
            search = GoogleSearch({
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "num": 5,
                "hl": "zh-cn",
            })
            results = search.get_dict()
            organic_results = results.get("organic_results", [])

            items = []
            for r in organic_results:
                items.append({
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                    "source": r.get("source", ""),
                })

            return json.dumps({
                "query": query,
                "count": len(items),
                "results": items,
            }, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {"error": f"搜索失败: {str(e)}"},
                ensure_ascii=False,
            )


# 供外部导入的工具实例
search_tool = SearchTool()
