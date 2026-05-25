"""App Store 评论抓取工具 —— 为 Agent 提供用户评论分析能力"""
import json
import logging
import time
import xml.etree.ElementTree as ET
from typing import ClassVar, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("AppStoreTool")

# 搜索回退地区列表（按优先级排列）
SEARCH_COUNTRIES = [
    ("cn", "中国区"),
    ("us", "美国区"),
    ("jp", "日本区"),
    ("gb", "英国区"),
    ("kr", "韩国区"),
]

# 常见 App 中文名 → App ID 硬编码映射（冷启动兜底）
KNOWN_APP_IDS = {
    "微信": (414478124, "微信"),
    "wechat": (414478124, "WeChat"),
    "抖音": (1142110895, "抖音"),
    "tiktok": (835599320, "TikTok"),
    "小红书": (741292507, "小红书"),
    "快手": (440948110, "快手"),
    "淘宝": (387682726, "淘宝"),
    "京东": (414245413, "京东"),
    "美团": (423084029, "美团"),
    "拼多多": (1044917946, "拼多多"),
    "支付宝": (333206289, "支付宝"),
    "滴滴": (554498813, "滴滴出行"),
    "bilibili": (736536022, "哔哩哔哩"),
    "百度": (382201985, "百度"),
    "微博": (350962117, "微博"),
}


class AppStoreToolInput(BaseModel):
    """App Store 评论抓取输入参数"""
    app_name: str = Field(..., description="App名称，支持中文")


class AppStoreTool(BaseTool):
    """抓取指定 App 在 App Store 的最新用户评论"""
    name: str = "fetch_app_reviews"
    description: str = (
        "抓取指定App在App Store的最新用户评论。"
        "输入App名称（支持中英文），自动在多个地区搜索匹配的App，"
        "返回最新评论（评分、标题、内容、作者、日期）。"
        "对主流中国App（微信、抖音、小红书等）有内置加速支持。"
    )
    args_schema: Type[BaseModel] = AppStoreToolInput

    SEARCH_URL: ClassVar[str] = "https://itunes.apple.com/search"

    @staticmethod
    def _search_app_id(term: str, country: str = "cn") -> tuple:
        """通过 iTunes Search API 搜索 App ID。返回 (app_id, name) 或 (None, None)。"""
        try:
            resp = requests.get(
                "https://itunes.apple.com/search",
                params={
                    "term": term,
                    "country": country,
                    "entity": "software",
                    "limit": 10,
                },
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None, None

            # 三层匹配：精确 > 包含 > 首条
            term_lower = term.lower()
            best = None
            for r in results:
                name = r.get("trackName", "")
                name_lower = name.lower()
                # 精确匹配
                if term_lower == name_lower:
                    best = r
                    break
                # 包含匹配
                if term_lower in name_lower or name_lower in term_lower:
                    if best is None:
                        best = r
            if best is None:
                best = results[0]

            return best.get("trackId"), best.get("trackName", term)
        except requests.exceptions.Timeout:
            return None, None
        except requests.exceptions.RequestException:
            return None, None
        except Exception:
            return None, None

    @staticmethod
    def _fetch_reviews(app_id, country: str = "cn") -> list:
        """通过 iTunes RSS 接口抓取评论（JSON + XML 双格式回退）。"""
        reviews = AppStoreTool._fetch_reviews_json(app_id, country)
        if not reviews:
            reviews = AppStoreTool._fetch_reviews_xml(app_id, country)
        return reviews

    @staticmethod
    def _fetch_reviews_json(app_id, country: str = "cn") -> list:
        """JSON 格式获取评论。"""
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"id={app_id}/sortBy=mostRecent/json"
        )
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            return []

        reviews = []
        for entry in entries:
            rating_info = entry.get("im:rating", {})
            if not rating_info:
                continue
            content = entry.get("content", {}).get("label", "")
            if not content or not content.strip():
                continue
            reviews.append({
                "rating": int(rating_info.get("label", 0)),
                "title": entry.get("title", {}).get("label", ""),
                "content": content,
                "author": entry.get("author", {}).get("name", {}).get("label", ""),
                "date": entry.get("updated", {}).get("label", ""),
            })
        return reviews

    @staticmethod
    def _fetch_reviews_xml(app_id, country: str = "cn") -> list:
        """XML 格式获取评论（JSON 不可用时的降级方案）。"""
        # Apple RSS XML 命名空间
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "im": "http://itunes.apple.com/rss",
        }
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"id={app_id}/sortBy=mostRecent/page=1/xml"
        )
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception:
            return []

        entries = root.findall("atom:entry", ns)
        if not entries:
            return []

        reviews = []
        for entry in entries:
            rating_el = entry.find("im:rating", ns)
            if rating_el is None:
                continue
            title_el = entry.find("atom:title", ns)
            content_el = entry.find("atom:content", ns)
            author_el = entry.find("atom:author/atom:name", ns)
            updated_el = entry.find("atom:updated", ns)

            content = (
                content_el.text.strip()
                if content_el is not None and content_el.text
                else ""
            )
            if not content:
                continue

            reviews.append({
                "rating": int(rating_el.text or 0),
                "title": title_el.text if title_el is not None else "",
                "content": content,
                "author": author_el.text if author_el is not None else "",
                "date": updated_el.text if updated_el is not None else "",
            })
        return reviews

    def _run(self, app_name: str) -> str:
        clean_name = app_name.strip()[:50]

        # ---- 0. 硬编码兜底：已知 App 直接使用预存 ID ----
        if clean_name.lower() in KNOWN_APP_IDS:
            app_id, matched_name = KNOWN_APP_IDS[clean_name.lower()]
            logger.info(f"使用预存 ID: {clean_name} → {app_id}")
        else:
            app_id = None
            matched_name = None

        # ---- 1. 多地区搜索（如无硬编码命中） ----
        if not app_id:
            for country, country_name in SEARCH_COUNTRIES:
                try:
                    app_id, matched_name = AppStoreTool._search_app_id(
                        clean_name, country
                    )
                    if app_id:
                        logger.info(
                            f"在{country_name}找到: {clean_name} → "
                            f"{matched_name} (ID: {app_id})"
                        )
                        break
                    time.sleep(0.4)
                except Exception:
                    time.sleep(0.4)
                    continue

        if not app_id:
            return json.dumps({
                "error": (
                    f"在 {len(SEARCH_COUNTRIES)} 个地区 App Store 均未找到"
                    f"与「{clean_name}」匹配的 App。"
                    "可能原因：1) 该产品无 iOS 版本；2) 名称较生僻，请尝试英文名或简称；"
                    "3) iTunes API 暂时不可用。"
                ),
                "suggestion": "请跳过 App Store 评论分析，基于市场搜索信息完成任务。",
            }, ensure_ascii=False)

        # ---- 2. 抓取评论（多地区 + 备选 ID 回退） ----
        reviews = []
        fetch_countries = ["cn", "us", "jp", "gb"]
        for fetch_country in fetch_countries:
            try:
                reviews = AppStoreTool._fetch_reviews(app_id, fetch_country)
                if reviews:
                    break
            except Exception:
                continue

        # 如果预存 ID 在全部地区都返回空，尝试重新搜索替代 ID
        if not reviews and clean_name.lower() in KNOWN_APP_IDS:
            for country, country_name in SEARCH_COUNTRIES:
                try:
                    alt_id, alt_name = AppStoreTool._search_app_id(clean_name, country)
                    if alt_id and alt_id != app_id:
                        alt_reviews = AppStoreTool._fetch_reviews(alt_id, country)
                        if alt_reviews:
                            app_id, matched_name = alt_id, alt_name
                            reviews = alt_reviews
                            logger.info(
                                f"预存 ID 无效，在{country_name}找到替代: "
                                f"{alt_name} (ID: {alt_id})"
                            )
                            break
                    time.sleep(0.3)
                except Exception:
                    time.sleep(0.3)
                    continue

        if not reviews:
            return json.dumps({
                "error": (
                    f"已找到 App「{matched_name}」（ID: {app_id}），"
                    f"但多地区 RSS 接口均返回空。Apple 可能限制了该 App 的评论公开访问。"
                ),
                "suggestion": (
                    "请使用 search_internet 工具搜索「{app_name} 用户评价」"
                    "从第三方平台获取用户反馈替代。"
                ),
                "app_name": matched_name,
                "app_id": app_id,
            }, ensure_ascii=False)

        return json.dumps({
            "app_name": matched_name or clean_name,
            "app_id": app_id,
            "country": "cn",
            "count": len(reviews),
            "reviews": reviews,
        }, ensure_ascii=False)


# 供外部导入的工具实例
app_store_tool = AppStoreTool()
