# PathAgent

PathAgent 是一个面向全切片病理图像（Whole-Slide Image，WSI）的研究型智能体框架。它使用可切换的病理图文 Retriever（PLIP 或 CONCH v1）检索候选区域、Patho-R1 生成可见形态学描述，再由文本大模型根据当前证据决定继续检索、观察、放大、聚焦或结束推理。

本仓库基于论文《PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning》的公开实现扩展，增加了原始 WSI 金字塔读取、多轮动作协议、结构化 Trace、证据引用校验、可恢复运行以及本地/远程 Executor 后端。

> 本项目仅用于科研和工程验证，不是医疗器械，不能替代病理医师诊断。

安装、输入格式、运行方法、Trace 审计、隐私安全和已知限制见 [使用与发布说明](USAGE.zh-CN.md)。

## 1. 项目定位和主要能力

PathAgent 通过协调检索模型、病理多模态模型和文本大模型，将单步 WSI 问答扩展为可追踪的多轮证据收集过程。

- 支持原始 WSI 和历史 JPEG patch 两种证据后端；
- 使用统一 Retriever 接口，以 PLIP 或 CONCH v1 完成问题相关候选区域检索；
- 使用 Patho-R1 生成局部、纯形态学观察；
- 支持 Qwen 本地 Transformers、OpenAI-compatible 服务和 DeepSeek 文本 Executor；
- 支持 `retrieve`、`inspect`、`zoom`、`focus`、`answer` 五类动作；
- 记录模型调用、动作、坐标、证据引用和终止状态等结构化 Trace；
- 可选确定性证据合同，用于检查证据充分性和引用可达性；
- 提供 Trace 审计、训练候选导出和本地 Trace Viewer。

## 2. PathAgent 架构与数据流

```text
问题 + WSI/patch 资产
        │
        ▼
Navigator（PLIP / CONCH v1 文图检索）
        │ 候选区域及相似度
        ▼
Environment（WSI 金字塔或 JPEG patch）
        │ 图像、倍率、Level-0 坐标
        ▼
Perceptor（Patho-R1）
        │ 可见形态学描述
        ▼
Executor（Qwen / DeepSeek）
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

## 3. 核心模块

| 路径 | 作用 |
|---|---|
| `pathagent.py` | 命令行入口、模型加载和协议分发 |
| `pathagent_v2.py` | 多轮 Agent 状态机、动作执行和证据状态维护 |
| `models/inference.py` | Patho-R1 与 Executor 提示词、输出清洗和解析 |
| `models/llm_backend.py` | Qwen Transformers、OpenAI-compatible 和 DeepSeek 后端 |
| `models/retrievers/` | 通用 Retriever 协议、PLIP 兼容适配器与 CONCH v1 适配器 |
| `models/evidence_contract.py` | 确定性证据合同和候选答案门控 |
| `models/trace_recorder.py` | 结构化 Trace、事件日志和盲法检查 |
| `patho_lora_sft/` | 形态学 Schema SFT、隐私门禁、患者隔离切分、盲审和受约束生成工具包 |
| `data_processing/wsi_pyramid.py` | WSI 金字塔读取、坐标换算、倍率选择和 focus 排序 |
| `scripts/` | 数据准备、运行、审计和结果处理工具 |
| `trace_viewer/` | Trace 的静态浏览界面 |
| `tests/` | 协议、证据、WSI、Trace 和恢复流程测试 |

SFT 工具包只发布方法代码、配置模板和合成测试，不包含患者数据、patch、教师原始响应、训练 JSONL、adapter 或运行记录。安装和安全边界见 [`patho_lora_sft/README.md`](patho_lora_sft/README.md)。
