# PathAgent 使用与发布说明

本文档说明 PathAgent 的环境依赖、输入格式、运行方法、Trace 审计、数据安全和已知限制。

## 环境与版本

推荐在 Linux 和 NVIDIA GPU 环境运行。当前验证过的主环境如下：

| 组件 | 版本或要求 |
|---|---|
| Python | 3.9 |
| PyTorch | 2.7.1 + CUDA 12.8 |
| torchvision | 0.22.1 + CUDA 12.8 |
| transformers | 4.51.0（PathAgent 主环境） |
| OpenSlide Python | 1.4.2 |
| h5py | 3.14.0 |
| PLIP | 外部源码仓库和对应 checkpoint |
| CONCH v1（可选） | 官方源码、用户自行获权下载的 checkpoint；仅限其许可允许的非商业科研用途 |
| Patho-R1 | Qwen2.5-VL 架构的 Patho-R1 checkpoint |
| Qwen3.5 | 建议在独立 Python 3.11 环境中启动服务 |

显存需求取决于模型大小、是否在线运行 Patho-R1 以及部署方式。建议将 Qwen Executor 独立部署为 OpenAI-compatible 服务，避免与 Patho-R1、Retriever 同进程竞争显存。

### 系统依赖

Ubuntu/Debian 可安装：

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 openslide-tools
```

### 安装主环境

```bash
conda create -n pathagent python=3.9 -y
conda activate pathagent

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如果本机 CUDA 版本不同，请从 PyTorch 官方安装说明中选择匹配的 wheel。

PLIP 当前以外部源码目录的形式加载：

```bash
git clone https://github.com/PathologyFoundation/plip.git /path/to/plip
```

模型权重、WSI、特征文件和数据集不包含在本仓库中，需要单独准备。

CONCH v1 也从官方源码目录加载。先按上游要求单独取得源码和 gated 权重，再安装兼容依赖：

```bash
pip install -r requirements-conch-v1.txt
```

本仓库不下载、不提交也不重新分发 CONCH 权重。当前适配器冻结官方 `conch_ViT-B-16` 的 448×448 输入、512维 contrastive projection、FP32推理及单位归一化输出；CONCH 第一版只支持原始 WSI 后端，不支持 `legacy_jpeg`。

### Qwen3.5 独立服务环境

主环境固定的 `transformers==4.51.0` 不支持当前 Qwen3.5 架构，因此建议单独创建 Python 3.11 环境。`requirements-qwen35-transformers.txt` 已固定到本项目验证过的 Transformers 提交。

```bash
uv venv .venv-qwen35 --python 3.11
uv pip install --python .venv-qwen35/bin/python \
  -r requirements-qwen35-transformers.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128

.venv-qwen35/bin/python scripts/qwen35_openai_server.py \
  --qwen_ckpt /path/to/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port 18004 \
  --model_name Qwen/Qwen3.5-4B
```

健康检查：

```bash
curl http://127.0.0.1:18004/health
```

## 输入数据

仓库不提供患者数据、WSI、报告、问题集、模型权重或预计算特征。以下示例全部使用虚构标识。

### 问题文件

`local_wsi` 支持 JSON 或 CSV。JSON 可以是数组，也可以使用 `questions` 包装：

```json
{
  "questions": [
    {
      "slide_id": "demo_slide_001",
      "question_id": "Q001",
      "question": "当前可见区域中最主要的结构模式是什么？",
      "choices": ["选项A", "选项B", "选项C", "证据不足"],
      "answer": ""
    }
  ]
}
```

正式盲法推理时，不要把金标准答案、报告证据或审核备注写入 Executor 可见输入。建议将标签保存在独立文件中，仅在推理结束后评分。

### WSI manifest

WSI 后端读取 JSONL；每行至少包含：

```json
{"slide_id":"demo_slide_001","slide_path":"/path/to/demo_slide_001.svs"}
```

### Patch manifest

每张 WSI 对应一个 JSONL，文件名建议为 `<slide_id>.jsonl`。选中区域至少需要：

```json
{"slide_id":"demo_slide_001","patch_id":"patch_0001","selected":true,"x_level0":0,"y_level0":0,"width_level0":4096,"height_level0":4096,"mpp_x":0.25,"mpp_y":0.25}
```

坐标必须使用 Level-0 坐标。不要在清单中写入姓名、住院号或其他直接身份信息。

### Retriever HDF5

PLIP 历史文件名为 `<slide_id>.plip.v1.h5`。CONCH v1 文件名为 `<slide_id>.conch_v1.h5`，并额外绑定 checkpoint、源码 revision、模型配置、预处理、patch manifest 和 WSI 指纹。每张 WSI 至少包含：

- 文件属性 `status="complete"`；
- 一维数据集 `patch_id`；
- 二维数据集 `features`，行数与 `patch_id` 完全一致。

CONCH 特征必须先由同一个 `pathagent` 环境预计算，不能拿 PLIP H5 与 CONCH 在线 query 混用：

```bash
conda run --no-capture-output -n pathagent python \
  scripts/precompute_retriever_features.py \
  --retriever-backend conch_v1 \
  --retriever-lib-path /path/to/conch/source \
  --retriever-checkpoint /path/to/conch/pytorch_model.bin \
  --wsi-manifest "${DATA_ROOT}/wsi_manifest.jsonl" \
  --patch-manifest-dir "${DATA_ROOT}/patch_manifests" \
  --output-dir "${DATA_ROOT}/retriever_features/conch_v1" \
  --device cuda:0 --precision fp32 --batch-size 16 \
  --run-name conch-v1-precompute-001
```

脚本逐 WSI 加锁、写唯一临时文件、关闭后重开自检并原子落盘。已有目标文件只有在全部 metadata、patch 顺序、维度、有限值和范数完全匹配时才会跳过；不匹配时停止，不会覆盖。

## 使用方法

先设置本地路径。以下变量仅存在于当前 shell，不要把真实数据路径写入版本库：

```bash
export PLIP_REPO=/path/to/plip
export PLIP_CKPT=/path/to/plip-checkpoint
export PATHO_R1_CKPT=/path/to/Patho-R1-7B
export QWEN_CKPT=/path/to/Qwen3.5-4B
export DATA_ROOT=/path/to/private-data
export RUN_ROOT=/path/to/output/pathagent-demo
```

### 使用原始 WSI 后端

```bash
python pathagent.py \
  --executor_protocol general_v2 \
  --zoom_backend wsi \
  --plip_lib_path "${PLIP_REPO}" \
  --plip_ckpt "${PLIP_CKPT}" \
  --patho_r1_ckpt "${PATHO_R1_CKPT}" \
  --executor_provider qwen \
  --qwen_ckpt "${QWEN_CKPT}" \
  --qwen_backend openai_compatible \
  --qwen_api_base_url http://127.0.0.1:18004/v1 \
  --qwen_api_model Qwen/Qwen3.5-4B \
  --wsi_manifest "${DATA_ROOT}/wsi_manifest.jsonl" \
  --patch_manifest_dir "${DATA_ROOT}/patch_manifests" \
  --feature_h5_dir "${DATA_ROOT}/plip_features" \
  --questions_file "${DATA_ROOT}/questions.json" \
  --dataset_name local_wsi \
  --save_dir "${RUN_ROOT}/results" \
  --trace_dir "${RUN_ROOT}/traces" \
  --run_id demo-run-001
```

`--zoom_backend wsi` 不会静默回退到 JPEG。缺少 WSI manifest、patch manifest 或 HDF5 特征时，程序会直接停止。

### 使用 CONCH v1 Retriever

在上面的 WSI 命令中，把三个 PLIP/feature 参数替换为通用参数：

```text
--retriever_backend conch_v1 \
--retriever_lib_path /path/to/conch/source \
--retriever_checkpoint /path/to/conch/pytorch_model.bin \
--retriever_feature_dir "${DATA_ROOT}/retriever_features/conch_v1"
```

PLIP 仍是默认后端，旧的 `--plip_lib_path`、`--plip_ckpt` 和 `--feature_h5_dir` 命令保持兼容。若新旧 PLIP 参数同时给出但路径冲突，程序会立即停止。CONCH 必须显式提供全部通用参数，不能从 PLIP 参数猜测路径。

### 使用历史 JPEG patch 后端

```bash
python pathagent.py \
  --executor_protocol general_v2 \
  --zoom_backend legacy_jpeg \
  --plip_lib_path "${PLIP_REPO}" \
  --plip_ckpt "${PLIP_CKPT}" \
  --patho_r1_ckpt "${PATHO_R1_CKPT}" \
  --executor_provider qwen \
  --qwen_ckpt "${QWEN_CKPT}" \
  --qwen_backend openai_compatible \
  --qwen_api_base_url http://127.0.0.1:18004/v1 \
  --qwen_api_model Qwen/Qwen3.5-4B \
  --descriptions_file "${DATA_ROOT}/patch_descriptions.json" \
  --feature_dir "${DATA_ROOT}/patch_features" \
  --patch_root "${DATA_ROOT}/patch_images" \
  --questions_file "${DATA_ROOT}/questions.json" \
  --dataset_name local_wsi \
  --save_dir "${RUN_ROOT}/results"
```

### 使用 DeepSeek Executor

DeepSeek 只替换文本 Executor；Retriever 与 Patho-R1 仍在本地运行。复制示例环境文件并限制权限：

```bash
cp api.env.example api.env
chmod 600 api.env
```

在 `api.env` 中填写：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

然后在 WSI 命令中改用：

```text
--executor_provider deepseek --executor_env_file api.env
```

不要在命令行中直接传递真实密钥；命令行参数可能出现在 shell 历史和进程列表中。调用外部 API 前，必须确认发送内容已经去标识化，并符合数据使用与伦理审批要求。

### 启用确定性证据合同

`general_v2` 可以用 `contract_v1` 代替模型自判证据充分性：

```text
--evidence_policy contract_v1 \
--evidence_contracts_path /path/to/evidence_contracts.json \
--option_ontology_path /path/to/option_ontology.json \
--descriptions_file /path/to/clean_descriptions.json \
--description_manifest /path/to/description_manifest.jsonl
```

该模式要求 WSI 后端、经过审核的描述 manifest 和可追溯证据引用。

## Trace 审计

运行结束后可检查 Trace 数量、动作、引用和终止状态：

```bash
python scripts/audit_trace_run.py \
  --trace_dir "${RUN_ROOT}/traces" \
  --expected_traces 1 \
  --expected_rollouts_per_question 1
```

运行输出可能包含 slide 标识、局部路径和派生证据。公开分享前必须再次脱敏，不要直接提交 `results/`、`traces/` 或本地审核页面。

## 测试

```bash
python -m pip install "pytest>=8,<10"
python -m pytest -q
```

默认单元测试不下载模型，也不会启动外部 API 正式调用；真实 CONCH smoke 需要用户自行准备的官方权重和本地 GPU。

## 可选工具依赖

部分数据准备脚本需要额外环境，例如 OpenCV、`segmentation-models-pytorch`、Trident、CLAM、Quilt-LLaVA 或 COCO caption evaluation。它们不属于核心推理依赖，应按照相应上游项目单独安装。不要把第三方源码、模型权重或数据集直接复制进本仓库。

## 数据隐私与发布安全

- `api.env`、`.env*`、模型权重、WSI、HDF5/NumPy 特征、运行结果和缓存不应进入 Git；
- 公开前应同时扫描当前文件和完整 Git 历史，仅删除最新版本中的敏感文件是不够的；
- WSI 路径、slide ID、患者 ID、报告文本、Trace 和审核记录均可能构成敏感研究数据；
- 外部 API 仅适合接收经过授权和去标识化的文本；
- 如果密钥曾经进入 Git 历史，应先吊销并重新生成，再重写历史。

## 已知限制

- 当前实现以研究脚本为主，尚未封装为可安装的 Python 包；
- 原始 WSI 资产构建依赖多个外部项目，尚未提供统一的一键下载流程；
- `pathagent_v2.py` 和部分实验编排脚本职责较多，后续适合拆分为状态、动作、策略和 I/O 子模块；
- 模型置信度尚未经过临床校准；有限 patch 上未发现证据不能推出整张 WSI 阴性；
- 本项目仅用于科研和工程验证，不能替代病理医师。

## 引用

```bibtex
@inproceedings{chen2026pathagent,
  title={Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning},
  author={Jingyun Chen and Linghan Cai and Zhikang Wang and Yi Huang and Songhan Jiang and Shenjin Huang and Hongpeng Wang and Yongbing Zhang},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

- 项目主页：[GitHub](https://github.com/G14nTDo4/PathAgent)
- 论文：[arXiv:2511.17052](https://arxiv.org/abs/2511.17052)

## License

本项目采用 Apache License 2.0，详见 `LICENSE`。
