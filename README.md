# PathAgent

PathAgent 是一个面向全切片病理图像（Whole-Slide Image，WSI）的研究型智能体框架。它使用 PLIP 检索候选区域、Patho-R1 生成可见形态学描述，再由文本大模型决定继续检索、观察、放大、聚焦或结束推理。

本仓库基于论文《PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning》的公开实现扩展，当前仅保留可运行的推理核心、自包含测试和必要示例配置。

> 本项目仅用于科研和工程验证，不是医疗器械，不能替代病理医师诊断。

安装、输入格式、运行方法、安全边界和已知限制见 [使用说明](USAGE.zh-CN.md)。

## 1. 项目定位和主要能力

- 支持原始 WSI 和历史 JPEG patch 两种证据后端；
- 使用 PLIP 完成问题相关的候选区域检索；
- 使用 Patho-R1 生成局部、纯形态学观察；
- 支持 Qwen Transformers、OpenAI-compatible 服务和 DeepSeek Executor；
- 支持 `retrieve`、`inspect`、`zoom`、`focus`、`answer` 等多轮动作；
- 记录模型调用、动作、坐标、证据引用和终止状态等结构化 Trace；
- 提供确定性证据合同，用于检查证据充分性和引用可达性。

## 2. PathAgent 架构与数据流

```text
问题 + WSI/patch 资产
        │
        ▼
Navigator（PLIP 文图检索）
        │ 候选区域及相似度
        ▼
Environment（WSI 金字塔或 JPEG patch）
        │ 图像、倍率、Level-0 坐标
        ▼
Perceptor（Patho-R1）
        │ 可见形态学描述
        ▼
Executor（Qwen / DeepSeek / OpenAI-compatible）
        │ 下一动作或候选答案
        ▼
Evidence Policy（模型判断或确定性合同）
        │
        ├── 证据不足：继续 retrieve / inspect / zoom / focus
        └── 证据充分：answer
                         │
                         ▼
                Result + Structured Trace
```

## 3. 核心模块说明

| 路径 | 作用 |
|---|---|
| `pathagent.py` | 命令行入口、模型加载和协议分发 |
| `pathagent_v2.py` | 多轮 Agent 状态机、动作执行和证据状态维护 |
| `models/inference.py` | Patho-R1 与 Executor 提示词、输出清洗和解析 |
| `models/llm_backend.py` | Qwen Transformers、OpenAI-compatible 和 DeepSeek 后端 |
| `models/evidence_contract.py` | 确定性证据合同和候选答案门控 |
| `models/trace_recorder.py` | 结构化 Trace 和盲法输入检查 |
| `models/retrieval_policy.py` | 初始检索和后续补充检索数量策略 |
| `data_processing/utils.py` | 数据读取、描述整理和坐标工具 |
| `data_processing/wsi_pyramid.py` | WSI 金字塔读取、坐标换算、倍率选择和 focus 排序 |
| `scripts/qwen35_openai_server.py` | 本地 Qwen3.5 OpenAI-compatible 服务入口 |
| `configs/` | 脱敏的证据合同与选项本体示例 |
| `tests/` | 不依赖模型、私有 WSI 或 API 的核心回归测试 |
