"""竞品分析Agent —— Streamlit 主入口"""
import datetime
import json
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, LLM, Task

from agents import market_analyst, sentiment_analyst, report_writer
from tasks import market_research_task, sentiment_analysis_task, report_writing_task

# ============================================================
# 兼容性处理
# ============================================================
# st.container(border=True) 需要 Streamlit >= 1.35
_STREAMLIT_HAS_BORDER = tuple(int(x) for x in st.__version__.split(".")) >= (1, 35)

def bordered_container():
    """返回带边框的容器（自动兼容旧版 Streamlit）。"""
    if _STREAMLIT_HAS_BORDER:
        return st.container(border=True)
    else:
        return st.container()

# 反馈数据存储路径
FEEDBACK_DIR = Path(__file__).parent / "data"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.json"

load_dotenv()

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="竞品分析Agent | 多智能体协作系统",
    page_icon="🔍",
    layout="wide",
)

# ============================================================
# 页面标题区域
# ============================================================
st.title("🔍 竞品分析Agent | 多智能体协作系统")
st.markdown("输入产品名，AI Agent 团队自动完成深度竞品分析——支持单品深度、竞品对比、口碑速览三种模式")
st.caption(
    "市场研究分析师 · 用户舆情分析师 · 产品战略报告撰写人  |  "
    "多源数据采集（搜索引擎 + 应用商店）|  可定制分析深度与关注维度"
)

# ============================================================
# 侧边栏 — API 配置
# ============================================================
st.sidebar.header("⚙️ API 配置")

deepseek_api_key = st.sidebar.text_input(
    "DeepSeek API Key",
    type="password",
    value=os.getenv("DEEPSEEK_API_KEY", ""),
)

serpapi_api_key = st.sidebar.text_input(
    "SerpAPI Key",
    type="password",
    value=os.getenv("SERPAPI_API_KEY", ""),
)

st.sidebar.divider()

# ============================================================
# 侧边栏 — 模型选择
# ============================================================
st.sidebar.header("🧠 模型选择")

model_name = st.sidebar.selectbox(
    "选择模型",
    options=["deepseek-v4-pro"],
    index=0,
    help="DeepSeek-V4-Pro：650B+ MoE架构，128K上下文，支持复杂推理与长文本理解",
)
st.sidebar.caption("deepseek-v4-pro（推荐）")

st.sidebar.divider()

# ============================================================
# 侧边栏 — 分析历史
# ============================================================
st.sidebar.header("📜 分析历史")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if st.session_state.analysis_history:
    for i, record in enumerate(st.session_state.analysis_history):
        st.sidebar.markdown(
            f"**{record['product']}**  "
            f"<small>{record['time']}</small>",
            unsafe_allow_html=True,
        )
else:
    st.sidebar.caption("暂无分析记录")

st.sidebar.divider()

# ============================================================
# 侧边栏 — 评分统计
# ============================================================
st.sidebar.header("📈 历史评分")

feedback_stats = {"avg": None, "count": 0}
if FEEDBACK_FILE.exists():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            all_feedback = json.load(f)
        if all_feedback:
            ratings = [fb["rating"] for fb in all_feedback]
            feedback_stats["avg"] = sum(ratings) / len(ratings)
            feedback_stats["count"] = len(ratings)
    except Exception:
        pass

if feedback_stats["count"] > 0:
    stars = "⭐" * round(feedback_stats["avg"])
    st.sidebar.markdown(f"{stars} **{feedback_stats['avg']:.1f} / 5**")
    st.sidebar.caption(f"共 {feedback_stats['count']} 条评分")
else:
    st.sidebar.caption("暂无评分")

st.sidebar.divider()

# ============================================================
# 侧边栏 — 关于本项目
# ============================================================
with st.sidebar.expander("ℹ️ 关于本项目"):
    st.markdown(
        """
        **多智能体协作的竞品分析系统**

        基于 CrewAI 框架构建，三个专业 Agent 分工协作，
        自动完成从信息搜集、舆情分析到报告撰写的全流程。

        **技术栈**
        - CrewAI（Agent 编排框架）
        - DeepSeek-V4-Pro（大语言模型）
        - Streamlit（Web 界面）
        - SerpAPI（互联网搜索）
        - App Store Scraper（评论抓取）

        **Agent 团队**
        - 🔍 市场研究分析师：搜集市场动态与行业信息
        - 📊 用户舆情分析师：分析 App Store 用户评论
        - 📝 报告撰写人：整合信息生成专业报告
        """
    )

st.sidebar.divider()

# ============================================================
# 侧边栏 — 使用说明
# ============================================================
with st.sidebar.expander("📖 使用说明"):
    st.markdown(
        """
        **步骤1**：配置 API Key（DeepSeek + SerpAPI）

        **步骤2**：输入要分析的产品名称

        **步骤3**：点击「开始分析」按钮

        **步骤4**：等待 Agent 团队协作完成分析

        **步骤5**：查看并下载分析报告
        """
    )

st.sidebar.divider()

# ============================================================
# 侧边栏 — 常见问题
# ============================================================
with st.sidebar.expander("❓ 常见问题"):
    st.markdown(
        """
        **Q: 提示网络连接超时**
        A: 检查网络连接，如使用代理请确认代理配置正确。

        **Q: 提示 API Key 无效**
        A: 请检查 DeepSeek 或 SerpAPI 的 Key 是否正确，
        确保没有多余空格。

        **Q: SerpAPI 搜索无结果**
        A: 可能月度免费额度已用完，请前往
        [SerpAPI 官网](https://serpapi.com) 检查账户余额。

        **Q: App Store 评论抓取失败**
        A: 系统会自动降级，基于其他信息继续生成报告。
        可尝试使用英文名称搜索。

        **Q: 分析耗时较长**
        A: 单次分析通常需要 30-90 秒，取决于搜索量
        和模型响应速度，请耐心等待。
        """
    )

# ============================================================
# 初始化 session_state
# ============================================================
if "report_content" not in st.session_state:
    st.session_state.report_content = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_product" not in st.session_state:
    st.session_state.current_product = ""
if "report_version" not in st.session_state:
    st.session_state.report_version = 1
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = ""
if "user_feedback_tags" not in st.session_state:
    st.session_state.user_feedback_tags = []
if "previous_report" not in st.session_state:
    st.session_state.previous_report = ""
# 分析参数（新增）
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "single"
if "analysis_depth" not in st.session_state:
    st.session_state.analysis_depth = "standard"
if "analysis_dimensions" not in st.session_state:
    st.session_state.analysis_dimensions = [
        "商业模式", "市场表现", "用户口碑", "产品功能",
        "定价策略", "技术架构", "营销策略", "版本迭代",
    ]
if "product_b" not in st.session_state:
    st.session_state.product_b = ""

# ============================================================
# 辅助函数
# ============================================================
def get_llm(api_key: str, model: str = "deepseek-v4-pro") -> LLM:
    """创建 DeepSeek LLM 实例"""
    return LLM(
        model=f"openai/{model}",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0.3,
    )


def extract_metrics(report: str) -> dict:
    """尝试从报告中提取关键指标，无法提取时返回默认值"""
    metrics = {
        "info_sources": "已完成",
        "review_count": "已完成",
        "suggestion_count": "已完成",
    }
    source_patterns = [
        r"(\d+)\s*个?\s*(信息源|来源|数据源|条信息|篇文章)",
        r"(信息源|来源|数据源)[：:]\s*(\d+)",
        r"搜集.*?(\d+)\s*条",
    ]
    for pattern in source_patterns:
        m = re.search(pattern, report)
        if m:
            num = m.group(1) if m.group(1).isdigit() else m.group(2)
            if num.isdigit():
                metrics["info_sources"] = f"{num} 个"
                break
    review_patterns = [
        r"(\d+)\s*条?\s*(评论|用户反馈|用户评价)",
        r"(评论|用户反馈|用户评价)[：:]\s*(\d+)",
        r"抓取.*?(\d+)\s*条",
        r"获取.*?(\d+)\s*条",
    ]
    for pattern in review_patterns:
        m = re.search(pattern, report)
        if m:
            num = m.group(1) if m.group(1).isdigit() else m.group(2)
            if num.isdigit():
                metrics["review_count"] = f"{num} 条"
                break
    suggestion_patterns = [
        r"(\d+)\s*条?\s*(战略|建议|启示|机会|策略)",
        r"(战略|建议|启示)[：:].*?(\d+)",
        r"提出.*?(\d+)\s*条",
    ]
    for pattern in suggestion_patterns:
        m = re.search(pattern, report)
        if m:
            num = m.group(1) if m.group(1).isdigit() else m.group(2)
            if num.isdigit():
                metrics["suggestion_count"] = f"{num} 条"
                break
    return metrics


def classify_error(error_msg: str) -> str:
    """根据错误信息分类并返回用户友好的提示"""
    msg_lower = error_msg.lower()
    if any(kw in msg_lower for kw in ["timeout", "timed out", "connect", "socket"]):
        return "网络连接超时，请检查网络后重试。如使用代理，请确认代理配置正确。"
    if any(kw in msg_lower for kw in ["401", "unauthorized", "invalid api key",
                                        "authentication", "key is invalid"]):
        return "API Key 无效，请检查后重新输入，确保没有多余空格。"
    if any(kw in msg_lower for kw in ["quota", "limit", "exceeded", "balance",
                                        "insufficient"]):
        return "搜索 API 额度可能已用完，请检查账户余额。"
    if any(kw in msg_lower for kw in ["app store", "appstore", "review", "scraper"]):
        return "App Store 评论抓取失败，系统将基于其他信息继续生成报告。"
    return f"分析过程中出现错误：{error_msg}"


# ============================================================
# 动态任务构建函数
# 根据分析模式、深度、维度，在运行时动态生成 Task 对象
# ============================================================
def build_dynamic_tasks(mode, product, product_b, depth_cfg, dimensions):
    """根据分析参数动态创建 Task 实例。返回 (tasks, agents, inputs)。"""
    dim_text = "、".join(dimensions) if dimensions else "全维度"
    depth_label = depth_cfg["label"]
    search_n = depth_cfg["search_num"]
    review_n = depth_cfg["review_num"]

    depth_dim_suffix = (
        f"\n\n**本次分析参数：**\n"
        f"- 分析深度：{depth_label}（搜索约 {search_n} 个信息源，"
        f"抓取约 {review_n} 条评论）\n"
        f"- 重点关注维度：{dim_text}\n"
        f"- 其他维度可简要提及或不涉及。"
    )

    if mode == "single":
        market_task = Task(
            description=market_research_task.description + depth_dim_suffix,
            expected_output=market_research_task.expected_output,
            agent=market_analyst,
            async_execution=True,
        )
        sentiment_task = Task(
            description=sentiment_analysis_task.description + depth_dim_suffix,
            expected_output=sentiment_analysis_task.expected_output,
            agent=sentiment_analyst,
            async_execution=True,
        )
        report_task = Task(
            description=report_writing_task.description + depth_dim_suffix,
            expected_output=report_writing_task.expected_output,
            agent=report_writer,
            context=[market_task, sentiment_task],
            markdown=True,
        )
        return ([market_task, sentiment_task, report_task],
                [market_analyst, sentiment_analyst, report_writer],
                {"product_name": product})

    elif mode == "compare":
        compare_header = f"本次为竞品对比分析模式，对比对象：**{product}** vs **{product_b}**。"
        market_task = Task(
            description=(
                f"针对 **{product}** 和 **{product_b}** 两个产品进行市场信息搜集。"
                f"请分别搜索两个产品的市场表现、商业策略、竞争格局等信息，"
                f"输出时对两个产品的市场表现进行初步对比。"
                f"{compare_header}{depth_dim_suffix}"
            ),
            expected_output="两个产品的市场信息对比摘要，分别列出各自的市场动态和策略要点。",
            agent=market_analyst,
            async_execution=True,
        )
        sentiment_task = Task(
            description=(
                f"分别获取 **{product}** 和 **{product_b}** 在 App Store 上的用户评论。"
                f"请分两次调用抓取工具，获取两个产品的评论数据，"
                f"输出时对两个产品的用户口碑进行对比。"
                f"{depth_dim_suffix}"
            ),
            expected_output="两个产品的用户口碑对比分析，分别列出各自的评分、好评Top3、差评Top3。",
            agent=sentiment_analyst,
            async_execution=True,
        )
        report_task = Task(
            description=(
                f"现在你收到了 **{product}** 和 **{product_b}** 的市场研究和用户口碑数据。\n\n"
                f"请撰写一份专业的竞品对比分析报告，结构如下：\n"
                f"## 一、{product} 概述（市场规模、核心优势、用户口碑摘要）\n"
                f"## 二、{product_b} 概述（市场规模、核心优势、用户口碑摘要）\n"
                f"## 三、多维度对比分析（用表格对比商业模式、市场表现、用户口碑、"
                f"产品功能、战略方向）\n"
                f"## 四、双方优劣势总结\n"
                f"## 五、战略启示（不少于4条可落地的建议）\n"
                f"## 六、风险提示\n\n"
                f"{depth_dim_suffix}"
            ),
            expected_output="专业的竞品对比分析报告，包含对比表格、优劣势分析和可落地的战略建议。",
            agent=report_writer,
            context=[market_task, sentiment_task],
            markdown=True,
        )
        return ([market_task, sentiment_task, report_task],
                [market_analyst, sentiment_analyst, report_writer],
                {"product_name": f"{product} vs {product_b}"})

    elif mode == "sentiment_only":
        sentiment_task = Task(
            description=(
                f"获取 **{product}** 在 App Store 上的最新用户评论（约 {review_n} 条），"
                f"进行用户口碑分析。\n\n"
                f"请直接输出以下内容（无需完整报告格式）：\n"
                f"1. **评分画像**：平均评分、各星级分布\n"
                f"2. **好评关键词**：用户最满意的3-5个方面\n"
                f"3. **差评关键词**：用户最不满的3-5个方面\n"
                f"4. **代表性评论**：好评和差评各摘录3条原文\n"
                f"{depth_dim_suffix}"
            ),
            expected_output="用户口碑速览报告，包含评分画像、好评/差评关键词和代表性评论。",
            agent=sentiment_analyst,
        )
        return ([sentiment_task],
                [sentiment_analyst],
                {"product_name": product})

    return None  # unreachable


# ============================================================
# 分析深度配置表
# ============================================================
DEPTH_CONFIG = {
    "quick":  {"label": "⚡ 快速扫描", "search_num": 3,  "review_num": 30},
    "standard": {"label": "🔍 标准分析", "search_num": 5,  "review_num": 50},
    "deep":   {"label": "🔬 深度研究", "search_num": 10, "review_num": 100},
}

# 预设场景模板
PRESET_TEMPLATES = {
    "custom":      {"label": "🎯 自定义",           "depth": "standard", "dims": None},
    "new_product":  {"label": "🆕 新产品立项调研",    "depth": "standard", "dims": ["商业模式", "市场表现", "用户口碑"]},
    "quarterly":    {"label": "📊 竞品季度监控",      "depth": "quick",    "dims": ["版本迭代", "营销策略", "市场表现"]},
    "ux_eval":      {"label": "👥 用户体验评估",      "depth": "deep",     "dims": ["用户口碑", "产品功能"]},
}

# ============================================================
# 主区域 — 分析配置面板
# ============================================================
input_container = st.container()
with input_container:
    st.subheader("🎯 分析配置")

    # -------- 预设场景模板 --------
    selected_preset = st.selectbox(
        "📋 场景模板",
        options=list(PRESET_TEMPLATES.keys()),
        format_func=lambda k: PRESET_TEMPLATES[k]["label"],
        key="preset_select",
    )
    # 应用模板默认值到 session_state
    template = PRESET_TEMPLATES[selected_preset]
    if selected_preset != "custom":
        st.session_state.analysis_depth = template["depth"]
        if template["dims"]:
            st.session_state.analysis_dimensions = template["dims"]

    # -------- 分析模式选择 --------
    st.markdown("##### 分析模式")
    analysis_mode = st.radio(
        "分析模式",
        options=["single", "compare", "sentiment_only"],
        format_func=lambda m: {
            "single": "🔍 单品深度分析",
            "compare": "⚔️ 竞品对比分析",
            "sentiment_only": "📊 用户口碑速览",
        }[m],
        horizontal=True,
        key="analysis_mode_radio",
        index=["single", "compare", "sentiment_only"].index(
            st.session_state.analysis_mode
        ),
    )
    st.session_state.analysis_mode = analysis_mode

    # 模式说明
    mode_descriptions = {
        "single": "完整的市场研究 + 用户口碑 + 战略报告，适合需要全面了解一个产品时使用。",
        "compare": "对两个竞品进行全方位对比分析，输出维度对比表和差异化战略建议。",
        "sentiment_only": "⚡ 快速模式，仅抓取和分析用户口碑，不执行市场搜索和完整报告撰写。",
    }
    st.caption(mode_descriptions[analysis_mode])

    # -------- 产品名称输入 --------
    if analysis_mode == "compare":
        col_a, col_b = st.columns(2)
        with col_a:
            product_name = st.text_input(
                "产品 A",
                placeholder="例如：抖音",
                key="product_a_input",
            )
        with col_b:
            product_b = st.text_input(
                "产品 B",
                placeholder="例如：快手",
                key="product_b_input",
            )
        st.session_state.product_b = product_b
    else:
        product_name = st.text_input(
            "请输入要分析的产品名称",
            placeholder="例如：抖音、TikTok、微信、小红书",
            key="product_name_input",
        )
        product_b = ""  # 非对比模式时不使用

    # -------- 高级选项（折叠面板） --------
    with st.expander("⚙️ 高级选项", expanded=False):
        # 分析深度
        st.markdown("##### 分析深度")
        analysis_depth = st.radio(
            "分析深度",
            options=["quick", "standard", "deep"],
            format_func=lambda d: DEPTH_CONFIG[d]["label"],
            horizontal=True,
            key="depth_radio",
            index=["quick", "standard", "deep"].index(
                st.session_state.analysis_depth
            ),
        )
        st.session_state.analysis_depth = analysis_depth
        depth_cfg = DEPTH_CONFIG[analysis_depth]
        st.caption(
            f"搜索约 {depth_cfg['search_num']} 个信息源，"
            f"抓取约 {depth_cfg['review_num']} 条评论"
        )

        # 关注维度
        st.markdown("##### 关注维度")
        all_dimensions = [
            "🏢 商业模式", "📈 市场表现", "👥 用户口碑", "🎯 产品功能",
            "💰 定价策略", "🔧 技术架构", "📣 营销策略", "🔄 版本迭代",
        ]
        # 将带 emoji 的选项映射回纯文本 key
        dim_key_map = {
            "🏢 商业模式": "商业模式", "📈 市场表现": "市场表现",
            "👥 用户口碑": "用户口碑", "🎯 产品功能": "产品功能",
            "💰 定价策略": "定价策略", "🔧 技术架构": "技术架构",
            "📣 营销策略": "营销策略", "🔄 版本迭代": "版本迭代",
        }
        default_dims = [
            k for k, v in dim_key_map.items()
            if v in st.session_state.analysis_dimensions
        ]
        selected_dims_display = st.multiselect(
            "选择本次分析重点关注的维度",
            options=list(dim_key_map.keys()),
            default=default_dims or list(dim_key_map.keys()),
            key="dims_multiselect",
        )
        # 存储纯文本维度
        analysis_dimensions = [dim_key_map[d] for d in selected_dims_display]
        st.session_state.analysis_dimensions = analysis_dimensions

    # -------- 开始分析按钮 --------
    start_button = st.button(
        "🚀 开始分析", type="primary", use_container_width=True
    )

# ============================================================
# 主区域 — 结果展示区域
# ============================================================
result_container = st.container()
with result_container:
    st.subheader("📊 分析结果")

    # 如果之前已完成分析，展示历史报告
    if st.session_state.analysis_done and st.session_state.report_content:
        # 显示版本标签
        version = st.session_state.get("report_version", 1)
        if version > 1:
            st.info(f"📝 基于反馈优化后的报告 v{version}")

        report = st.session_state.report_content

        # 锚点：返回顶部
        st.markdown('<div id="report-top"></div>', unsafe_allow_html=True)

        # 如果报告过长（>5000字符），分章节折叠显示
        if len(report) > 5000:
            sections = re.split(r"\n(?=#{1,3}\s)", report)
            if len(sections) > 1:
                # 摘要/首段直接展示
                intro = sections[0].strip()
                if intro:
                    st.markdown(intro)

                # 后续章节折叠显示
                for i, section in enumerate(sections[1:], start=1):
                    title_line = section.strip().split("\n")[0]
                    title_clean = title_line.lstrip("#").strip()
                    with st.expander(f"📄 {title_clean}", expanded=(i <= 2)):
                        st.markdown(section)
            else:
                st.markdown(report)
        else:
            st.markdown(report)

        # 返回顶部
        st.markdown(
            '<a href="#report-top" style="text-decoration:none;">'
            "⬆️ 返回顶部</a>",
            unsafe_allow_html=True,
        )

        # 下载按钮
        st.divider()
        version_tag = f"_v{version}" if version > 1 else ""
        st.download_button(
            label="📥 下载报告（Markdown）",
            data=st.session_state.report_content,
            file_name=(
                f"{st.session_state.current_product}_竞品分析报告"
                f"{version_tag}_{datetime.date.today().strftime('%Y%m%d')}.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )

        # ============================================
        # 评分与反馈区域（每次报告展示后均出现）
        # ============================================
        st.divider()
        with bordered_container():
            st.markdown("##### 📊 为这份报告评分")

            rating = st.select_slider(
                "评分",
                options=[1, 2, 3, 4, 5],
                value=4,
                format_func=lambda x: "⭐" * x,
                key=f"rating_v{version}",
            )
            st.caption(f"当前评分：{'⭐' * rating} ({rating}/5)")

            feedback_text = st.text_area(
                "💬 您对报告有什么建议？（选填）",
                placeholder=(
                    "例如：希望多分析商业模式、用户增长数据不够详细、"
                    "建议增加竞品对比维度..."
                ),
                key=f"feedback_text_v{version}",
            )

            feedback_tags = st.multiselect(
                "快速标签（可多选）",
                options=[
                    "信息不够全面",
                    "分析深度不足",
                    "缺少数据支撑",
                    "战略建议太笼统",
                    "报告结构不清晰",
                    "引用来源不够",
                    "分析很到位",
                    "内容超出预期",
                ],
                key=f"feedback_tags_v{version}",
            )

            # 提交评分按钮
            if st.button("✅ 提交评分", key=f"submit_feedback_v{version}"):
                # 保存反馈到本地 JSON
                FEEDBACK_DIR.mkdir(exist_ok=True)
                existing = []
                if FEEDBACK_FILE.exists():
                    try:
                        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    except Exception:
                        existing = []

                existing.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "product_name": st.session_state.current_product,
                    "report_version": version,
                    "rating": rating,
                    "feedback_text": feedback_text,
                    "feedback_tags": feedback_tags,
                    "report_preview": report[:200],
                })

                with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

                # 保存反馈内容供重新生成使用
                st.session_state.feedback_submitted = True
                st.session_state.user_feedback = feedback_text
                st.session_state.user_feedback_tags = feedback_tags
                st.session_state.previous_report = report

                st.toast("🎉 感谢您的反馈！我们会持续优化分析质量")
                st.rerun()

        # 基于反馈重新生成按钮（仅提交评分后显示）
        if st.session_state.feedback_submitted:
            st.divider()
            st.markdown("##### 🔄 优化报告")

            # 汇总用户反馈
            feedback_summary_parts = []
            if st.session_state.user_feedback:
                feedback_summary_parts.append(
                    f"用户建议：{st.session_state.user_feedback}"
                )
            if st.session_state.user_feedback_tags:
                feedback_summary_parts.append(
                    f"用户反馈标签：{'、'.join(st.session_state.user_feedback_tags)}"
                )
            feedback_summary = "；".join(feedback_summary_parts)

            st.caption(f"将基于以下反馈重新生成：_{feedback_summary}_")

            if st.button(
                "🔄 基于您的反馈重新生成报告",
                type="secondary",
                key=f"regenerate_v{version}",
            ):
                st.session_state.feedback_submitted = False
                st.session_state.report_version = version + 1

                with st.status(
                    "🤖 正在基于反馈重新生成报告...", expanded=True
                ) as regen_status:
                    st.markdown(f"**原始报告版本：** v{version}")
                    st.markdown(f"**用户反馈：** {feedback_summary}")
                    st.divider()

                    # 构建重新生成任务
                    regen_description = (
                        f"以下是针对产品「{st.session_state.current_product}」"
                        f"的原始竞品分析报告 v{version}。\n\n"
                        f"用户阅读后给出了以下反馈：\n"
                        f"**{feedback_summary}**\n\n"
                        f"请根据用户反馈，对报告进行针对性的改进和优化：\n"
                        f"1. 对用户指出的不足之处进行补充和深化\n"
                        f"2. 保留原报告中用户认可的部分\n"
                        f"3. 如果反馈中提到缺少数据支撑，搜索更多数据并引用\n"
                        f"4. 如果反馈中提到战略建议太笼统，"
                        f"补充更具体、可落地的行动方案\n\n"
                        f"---\n\n"
                        f"**原始报告内容：**\n\n"
                        f"{st.session_state.previous_report}"
                    )

                    regen_task = Task(
                        description=regen_description,
                        expected_output=(
                            "基于用户反馈优化后的完整 Markdown 格式竞品分析报告，"
                            "保留原报告结构，针对反馈意见进行针对性改进。"
                        ),
                        agent=report_writer,
                        markdown=True,
                    )

                    # 仅执行报告撰写任务
                    llm = get_llm(deepseek_api_key, model_name)
                    report_writer.llm = llm

                    regen_crew = Crew(
                        agents=[report_writer],
                        tasks=[regen_task],
                        verbose=True,
                    )

                    st.markdown("🔄 报告撰写人正在根据反馈优化报告...")

                    regen_start = time.time()
                    regen_result = regen_crew.kickoff()
                    regen_elapsed = time.time() - regen_start

                    st.markdown(
                        f"✅ 优化完成！耗时 {regen_elapsed:.0f} 秒"
                    )
                    regen_status.update(
                        label="✅ 优化完成！",
                        state="complete",
                        expanded=False,
                    )

                # 提取新报告
                new_report = (
                    regen_result.raw
                    if hasattr(regen_result, "raw")
                    else str(regen_result)
                )
                st.session_state.report_content = new_report

                st.rerun()


# ============================================================
# 按钮点击执行逻辑
# ============================================================
if start_button:
    # --- 输入验证（按模式区分） ---
    mode_label = {
        "single": "单品深度分析",
        "compare": "竞品对比分析",
        "sentiment_only": "用户口碑速览",
    }
    if analysis_mode == "compare":
        if not product_name.strip() or not product_b.strip():
            st.warning("竞品对比模式需要同时填写产品 A 和产品 B")
        else:
            product_name = product_name.strip()
            product_b = product_b.strip()
    elif not product_name.strip():
        st.warning("请输入产品名称")
    else:
        product_name = product_name.strip()

    # 二次检查（确保 product_b 始终有值）
    if product_b is None:
        product_b = ""

    # --- API Key 验证 ---
    if not deepseek_api_key:
        st.warning("请配置 DeepSeek API Key")
    elif not serpapi_api_key:
        st.warning("请配置 SerpAPI Key")
    elif not product_name.strip():
        pass  # 已在上面处理
    else:
        # --- 设置环境变量 ---
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        os.environ["SERPAPI_API_KEY"] = serpapi_api_key

        try:
            now_str = datetime.datetime.now().strftime
            depth_cfg = DEPTH_CONFIG[analysis_depth]

            # -------- 构建动态任务 --------
            task_result = build_dynamic_tasks(
                analysis_mode, product_name, product_b,
                depth_cfg, analysis_dimensions,
            )
            if task_result is None:
                st.error("无法构建分析任务，请检查分析参数配置。")
                st.stop()
            tasks, agents, crew_inputs = task_result
            has_market = any(a == market_analyst for a in agents)
            has_sentiment = any(a == sentiment_analyst for a in agents)
            has_report = any(a == report_writer for a in agents)

            # 提前计算展示名称（避免在 with 块内赋值导致作用域问题）
            display_target = (
                f"{product_name} vs {product_b}"
                if analysis_mode == "compare"
                else product_name
            )

            # ================================================
            # Agent 协作实时看板
            # ================================================
            with st.status(
                "🤖 Agent 团队协作中...", expanded=True
            ) as status:

                # -------- 区域1：任务总览 --------
                st.markdown(f"**分析目标：** {display_target}")
                st.caption(f"模式：{mode_label[analysis_mode]} | "
                           f"深度：{depth_cfg['label']}")
                phase_placeholder = st.empty()
                phase_placeholder.markdown("🚀 启动 Agent 协作系统...")

                st.divider()

                # -------- 区域2：Agent 工作状态（三列卡片） --------
                if analysis_mode == "sentiment_only":
                    st.caption("⚡ 快速模式：仅执行用户口碑分析")
                else:
                    st.caption(
                        "⚡ 市场研究分析师和用户舆情分析师正在并行工作"
                    )
                    st.caption(
                        "💡 前两个 Agent 并行执行，第三个 Agent "
                        "在前两个完成后自动启动"
                    )

                col1, col2, col3 = st.columns(3)

                with col1:
                    with bordered_container():
                        if has_market:
                            st.markdown("##### 🔍 市场研究分析师")
                            agent1_status = st.empty()
                            agent1_desc = st.empty()
                            agent1_status.markdown(
                                "<span style='color:#3b82f6;'>🔄 工作中</span>",
                                unsafe_allow_html=True,
                            )
                            agent1_desc.caption("正在搜索相关信息...")
                        else:
                            st.markdown("##### 🔍 市场研究分析师")
                            st.markdown(
                                "<span style='color:#9ca3af;'>⚫ 未启用</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption("快捷模式不执行市场搜索")

                with col2:
                    with bordered_container():
                        if has_sentiment:
                            st.markdown("##### 📊 用户舆情分析师")
                            agent2_status = st.empty()
                            agent2_desc = st.empty()
                            agent2_status.markdown(
                                "<span style='color:#3b82f6;'>🔄 工作中</span>",
                                unsafe_allow_html=True,
                            )
                            agent2_desc.caption("正在抓取 App Store 评论...")
                        else:
                            st.markdown("##### 📊 用户舆情分析师")
                            st.markdown(
                                "<span style='color:#9ca3af;'>⚫ 未启用</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption("未启用舆情分析")

                with col3:
                    with bordered_container():
                        if has_report:
                            st.markdown("##### 📝 报告撰写人")
                            agent3_status = st.empty()
                            agent3_desc = st.empty()
                            agent3_status.markdown(
                                "<span style='color:#9ca3af;'>⏳ 等待中</span>",
                                unsafe_allow_html=True,
                            )
                            agent3_desc.caption("等待前序任务完成...")
                        else:
                            st.markdown("##### 📝 报告撰写人")
                            st.markdown(
                                "<span style='color:#9ca3af;'>⚫ 未启用</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption("快捷模式直接输出分析结果")

                st.divider()

                # -------- 区域3：执行日志 --------
                log_expander = st.expander("📋 详细执行日志", expanded=False)
                log_lines = []

                # 初始日志（按实际启用的 Agent 生成）
                log_lines.append(
                    f"[{now_str('%H:%M:%S')}] 系统 | 模式={mode_label[analysis_mode]}，"
                    f"深度={depth_cfg['label']}，维度={len(analysis_dimensions)}个，"
                    f"任务调度完成"
                )
                if has_market:
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 🔍 市场研究分析师 | 开始搜索相关信息"
                    )
                if has_sentiment:
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 📊 用户舆情分析师 | 开始抓取 App Store 评论"
                    )
                log_expander.text("\n".join(log_lines))

                # 更新阶段状态
                phase_placeholder.markdown("📋 任务已分配，开始并行执行...")

                # ================================================
                # 初始化 LLM 并执行 Crew（阻塞调用）
                # ================================================
                llm = get_llm(deepseek_api_key, model_name)

                # 更新所有用到 Agent 的 LLM
                for agent in agents:
                    agent.llm = llm

                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=True,
                )

                start_time = time.time()
                result = crew.kickoff(inputs=crew_inputs)
                elapsed = time.time() - start_time

                # ================================================
                # 更新看板状态 —— 阶段1：前两个 Agent 完成，第三个启动
                # ================================================
                phase_placeholder.markdown(
                    "📝 正在生成最终报告..."
                    if has_report else "✅ 分析完成"
                )

                # 任务1+2 标记完成
                if has_market:
                    agent1_status.markdown(
                        "<span style='color:#22c55e;'>✅ 已完成</span>",
                        unsafe_allow_html=True,
                    )
                    agent1_desc.caption("已找到信息源，信息整合完成")

                if has_sentiment:
                    agent2_status.markdown(
                        "<span style='color:#22c55e;'>✅ 已完成</span>",
                        unsafe_allow_html=True,
                    )
                    agent2_desc.caption("已获取评论，情感分析完成")

                # 日志：前两个 Agent 完成
                if has_market:
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 🔍 市场研究分析师 | 搜索完成，信息整合完成"
                    )
                if has_sentiment:
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 📊 用户舆情分析师 | 评论抓取完成，情感分析完成"
                    )

                # 任务3 从"等待中"切换到"工作中"
                if has_report:
                    agent3_status.markdown(
                        "<span style='color:#3b82f6;'>🔄 工作中</span>",
                        unsafe_allow_html=True,
                    )
                    agent3_desc.caption("正在读取分析结果，撰写报告...")
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 📝 报告撰写人 | 开始整合分析结果，撰写报告"
                    )
                    log_expander.text("\n".join(log_lines))

                    # 短暂停留让用户看到过渡状态（模拟实时切换效果）
                    time.sleep(1.2)

                # ================================================
                # 更新看板状态 —— 阶段2：全部完成
                # ================================================
                if has_report:
                    agent3_status.markdown(
                        "<span style='color:#22c55e;'>✅ 已完成</span>",
                        unsafe_allow_html=True,
                    )
                    agent3_desc.caption("报告生成完成")
                    log_lines.append(
                        f"[{now_str('%H:%M:%S')}] 📝 报告撰写人 | 报告生成完成"
                    )

                log_lines.append(
                    f"[{now_str('%H:%M:%S')}] 系统 | 全部任务完成，总耗时 {elapsed:.0f} 秒"
                )
                log_expander.text("\n".join(log_lines))

                phase_placeholder.markdown("✅ 分析完成")
                status.update(
                    label="✅ 分析完成！",
                    state="complete",
                    expanded=False,
                )

            # --- 提取报告内容 ---
            report_content = result.raw if hasattr(result, "raw") else str(result)

            # --- 提取关键指标 ---
            metrics = extract_metrics(report_content)

            # --- 保存到 session_state ---
            st.session_state.report_content = report_content
            st.session_state.analysis_done = True
            st.session_state.current_product = product_name

            # --- 添加到历史记录（保留最近3条） ---
            history_entry = {
                "product": display_target,
                "time": datetime.datetime.now().strftime("%m-%d %H:%M"),
            }
            st.session_state.analysis_history.insert(0, history_entry)
            st.session_state.analysis_history = st.session_state.analysis_history[:3]

            # ================================================
            # 关键指标展示区（4列）
            # ================================================
            st.subheader("📈 分析概览")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ 分析耗时", f"{elapsed:.1f} 秒")
            col2.metric("🔗 搜索信息源", metrics["info_sources"])
            col3.metric("💬 分析评论数", metrics["review_count"])
            col4.metric("💡 战略建议数", metrics["suggestion_count"])

            st.divider()

            # ================================================
            # 展示报告（带折叠优化）
            # ================================================
            st.subheader("📊 分析结果")
            st.success(f"✅ 分析完成！耗时 {elapsed:.0f} 秒")

            # 锚点
            st.markdown('<div id="report-top"></div>', unsafe_allow_html=True)

            # 长报告分章节折叠
            if len(report_content) > 5000:
                sections = re.split(r"\n(?=#{1,3}\s)", report_content)
                if len(sections) > 1:
                    intro = sections[0].strip()
                    if intro:
                        st.markdown(intro)

                    for i, section in enumerate(sections[1:], start=1):
                        title_line = section.strip().split("\n")[0]
                        title_clean = title_line.lstrip("#").strip()
                        with st.expander(
                            f"📄 {title_clean}", expanded=(i <= 2)
                        ):
                            st.markdown(section)
                else:
                    st.markdown(report_content)
            else:
                st.markdown(report_content)

            # 返回顶部
            st.markdown(
                '<a href="#report-top" style="text-decoration:none;">'
                "⬆️ 返回顶部</a>",
                unsafe_allow_html=True,
            )

            # ================================================
            # 下载按钮
            # ================================================
            st.divider()
            mode_prefix = {
                "single": "",
                "compare": "对比",
                "sentiment_only": "口碑速览",
            }
            st.download_button(
                label="📥 下载报告（Markdown）",
                data=report_content,
                file_name=(
                    f"{product_name}{'_vs_' + product_b if product_b else ''}"
                    f"_{mode_prefix[analysis_mode]}分析报告_"
                    f"{datetime.date.today().strftime('%Y%m%d')}.md"
                ),
                mime="text/markdown",
                use_container_width=True,
            )

            # 刷新页面以更新历史结果区域
            st.rerun()

        except Exception as e:
            friendly_msg = classify_error(str(e))
            st.error(f"❌ {friendly_msg}")
            st.caption(
                "如需帮助，请查看侧边栏「常见问题」或检查 API Key 配置。"
            )

# ============================================================
# 页面底部
# ============================================================
st.divider()
st.caption("💡 本工具基于 AI 生成，分析结果仅供参考，不构成任何商业决策建议。")
st.caption(
    "技术栈：CrewAI + DeepSeek-V4-Pro + Streamlit + SerpAPI + App Store Scraper"
)
