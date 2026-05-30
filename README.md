# 🔍 AI竞品分析Agent - 多智能体协作系统

基于CrewAI的多智能体协作系统，输入产品名即可自动生成深度竞品分析报告。

## 功能特点

- **多智能体协作**：三个专业Agent分工完成市场研究、用户舆情分析和报告撰写
- **并行执行**：市场研究和舆情分析同时进行，效率提升50%
- **专业报告**：生成包含战略建议的结构化Markdown报告
- **可视化界面**：Streamlit构建的友好交互界面
- **低成本**：使用DeepSeek API，单次分析成本低于0.1元

## 项目架构

```
用户输入产品名称
      ↓
Streamlit 界面
      ↓
CrewAI 多智能体协作系统
      ↓
┌──────────┼──────────┐
↓          ↓          ↓
市场研究   用户舆情   报告撰写
分析师     分析师     Agent
↓          ↓          ↓
SerpAPI   App Store  综合分析
搜索      评论抓取    生成报告
      ↓
竞品分析报告输出
```

## 安装步骤

1. 克隆项目
2. 创建虚拟环境：`python -m venv venv`
3. 激活虚拟环境：
   - Windows：`venv\Scripts\activate`
   - macOS/Linux：`source venv/bin/activate`
4. 安装依赖：`pip install -r requirements.txt`
5. 配置环境变量：复制 `.env.example` 为 `.env` 并填入 API Key
6. 运行应用：`streamlit run app.py`

## 使用方法

1. 在侧边栏配置 DeepSeek API Key 和 SerpAPI Key（或写入 `.env` 文件自动读取）
2. 输入要分析的产品名称（如：抖音、TikTok、微信）
3. 点击「开始分析」按钮
4. 等待 Agent 团队协作完成分析（通常 30-90 秒）
5. 查看并下载完整的 Markdown 格式分析报告

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | CrewAI |
| 大语言模型 | DeepSeek-V4-Pro |
| 前端界面 | Streamlit |
| 互联网搜索 | SerpAPI |
| 评论抓取 | app-store-scraper |

## 项目结构

```
/
├── app.py                 # Streamlit 主入口
├── agents.py              # Agent 角色定义（3个专业Agent）
├── tasks.py               # 任务流定义（并行+串行编排）
├── tools/                 # 自定义工具
│   ├── __init__.py
│   ├── search_tool.py     # SerpAPI 搜索封装
│   └── app_store_tool.py  # App Store 评论抓取封装
├── utils.py               # 辅助函数（日志配置）
├── requirements.txt       # Python 依赖清单
├── .env.example           # 环境变量模板
└── README.md
```

## API Key 获取

- **DeepSeek**：https://platform.deepseek.com
- **SerpAPI**：https://serpapi.com

## 成本说明

单次分析约消耗 DeepSeek API 10K-20K tokens，成本约 0.05 元人民币。
SerpAPI 免费额度每月 100 次搜索。

## 许可证

MIT License
