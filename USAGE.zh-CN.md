# PathAgent 使用说明

本文档说明最小公开版本的环境依赖、输入格式、运行方法、Trace、安全边界和已知限制。

## 环境与版本

推荐使用 Linux 和 NVIDIA GPU。当前验证环境如下：

| 组件 | 版本或要求 |
|---|---|
| Python | 3.9 |
| PyTorch | 2.7.1 + CUDA 12.8 |
| torchvision | 0.22.1 + CUDA 12.8 |
| transformers | 4.51.0（主环境） |
| OpenSlide Python | 1.4.2 |
| h5py | 3.14.0 |
| PLIP | 外部源码和 checkpoint |
| Patho-R1 | Qwen2.5-VL 架构 checkpoint |
| Qwen3.5 服务 | 建议使用独立 Python 3.11 环境 |

安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y libopenslide0 openslide-tools
```

安装主环境：

```bash
conda create -n pathagent python=3.9 -y
conda activate pathagent

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

PLIP 当前以外部源码目录的形式加载：

```bash
git clone https://github.com/PathologyFoundation/plip.git /path/to/plip
```

模型权重、WSI、特征文件和数据集不包含在仓库中，需要单独准备。

### Qwen3.5 独立服务

主环境固定的 `transformers==4.51.0` 不支持当前 Qwen3.5 架构。可建立独立环境并启动 OpenAI-compatible 服务：

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

以下示例均使用虚构标识。仓库不提供患者数据、报告、WSI、模型权重或预计算特征。

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

盲法推理时不要把金标准答案、报告证据或审核备注写入 Executor 可见输入。

### WSI manifest

WSI 后端读取 JSONL，每行至少包含：

```json
{"slide_id":"demo_slide_001","slide_path":"/path/to/demo_slide_001.svs"}
```

### Patch manifest

每张 WSI 对应一个 `<slide_id>.jsonl`：

```json
{"slide_id":"demo_slide_001","patch_id":"patch_0001","selected":true,"x_level0":0,"y_level0":0,"width_level0":4096,"height_level0":4096,"mpp_x":0.25,"mpp_y":0.25}
```

坐标必须使用 Level-0 坐标。

### PLIP HDF5

每张 WSI 对应一个 `<slide_id>.h5`，至少包含：

- 文件属性 `status="complete"`；
- 一维数据集 `patch_id`；
- 二维数据集 `features`，行数与 `patch_id` 一致。

### GrandQC tissue mask

WSI focus 需要每张切片对应的二值组织 mask，文件名为 `<slide_id>.grandqc.png`。通过 `--focus_tissue_mask_dirs` 显式指定目录；程序不会搜索内部约定路径。

## 使用方法

先设置本地路径。不要把真实数据路径提交到 Git：

```bash
export PLIP_REPO=/path/to/plip
export PLIP_CKPT=/path/to/plip-checkpoint
export PATHO_R1_CKPT=/path/to/Patho-R1-7B
export QWEN_CKPT=/path/to/Qwen3.5-4B
export DATA_ROOT=/path/to/private-data
export RUN_ROOT=/path/to/output/pathagent-demo
```

### 原始 WSI 后端

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
  --focus_tissue_mask_dirs "${DATA_ROOT}/grandqc_masks" \
  --questions_file "${DATA_ROOT}/questions.json" \
  --dataset_name local_wsi \
  --save_dir "${RUN_ROOT}/results" \
  --trace_dir "${RUN_ROOT}/traces" \
  --run_id demo-run-001
```

`--zoom_backend wsi` 不会静默回退到 JPEG。缺少 WSI、patch manifest 或 HDF5 特征时程序会停止。

### 历史 JPEG patch 后端

将上面命令中的 WSI 参数替换为：

```text
--zoom_backend legacy_jpeg \
--descriptions_file "${DATA_ROOT}/patch_descriptions.json" \
--feature_dir "${DATA_ROOT}/patch_features" \
--patch_root "${DATA_ROOT}/patch_images"
```

### DeepSeek Executor

复制示例环境文件并限制权限：

```bash
cp api.env.example api.env
chmod 600 api.env
```

填写 `DEEPSEEK_API_KEY` 后，将运行参数改为：

```text
--executor_provider deepseek --executor_env_file api.env
```

不要在命令行直接传递真实密钥。调用外部 API 前，必须确认发送内容已经去标识化并符合数据使用审批。

### 确定性证据合同

仓库提供两份脱敏示例：

- `configs/vqa_evidence_contracts_v0.1.json`
- `configs/vqa_ontology_v0.1.json`

启用方式：

```text
--evidence_policy contract_v1 \
--evidence_contracts_path configs/vqa_evidence_contracts_v0.1.json \
--option_ontology_path configs/vqa_ontology_v0.1.json \
--descriptions_file /path/to/clean_descriptions.json \
--description_manifest /path/to/description_manifest.jsonl
```

该模式要求 WSI 后端、经过审核的描述 manifest 和可追溯证据引用。

## Trace 与测试

指定 `--trace_dir` 后，PathAgent 会写入事件日志和最终结构化 Trace。Trace 可能包含 slide 标识、局部路径和派生证据，分享前必须脱敏。

核心测试使用 mock、临时文件和合成数据，不需要模型权重、私有 WSI 或 API 密钥：

```bash
python -m pip install "pytest>=8,<10"
python -m pytest -q
```

## 数据隐私

- 不要提交 `api.env`、`.env*`、模型权重、WSI、HDF5/NumPy 特征、运行结果或缓存；
- WSI 路径、slide ID、患者 ID、报告文本和 Trace 都可能包含敏感信息；
- 外部 API 只应接收经过授权和去标识化的数据；
- 如果密钥曾进入 Git，应立即吊销，并在公开前重写 Git 历史。

## 已知限制

- 当前实现仍以研究代码为主，尚未封装为可安装的 Python 包；
- WSI、patch manifest 和 PLIP 特征需要用户自行生成；
- 模型置信度未经临床校准；有限 patch 上未发现证据不能推出整张 WSI 阴性；
- 本项目仅用于科研与工程验证，不能替代病理医师。

## 引用与许可证

```bibtex
@inproceedings{chen2026pathagent,
  title={Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning},
  author={Jingyun Chen and Linghan Cai and Zhikang Wang and Yi Huang and Songhan Jiang and Shenjin Huang and Hongpeng Wang and Yongbing Zhang},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

本项目采用 Apache License 2.0，详见 `LICENSE`。
