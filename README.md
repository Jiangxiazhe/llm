## 项目简介

这是一个从零**实现与理解大语言模型（LLM）完整训练链路**的实践项目，目标是打通：

- **自定义 tokenizer 训练**
- **从头预训练 Tiny 级别 Transformer 语言模型**
- **基于指令数据的监督微调（SFT）**
- **基于 HuggingFace/DeepSpeed 的工程化训练管线**
- **本地推理与采样 demo**

项目代码和实验会同步到 GitHub，方便复现和二次开发。

## 目录结构概览

- `tokenizer_code/`
  - `train_tokenizer.py`：从 `jsonl` 语料训练自定义 BPE tokenizer，并生成 `tokenizer_k/` 目录（含 `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json` 等），内置对话 `chat_template`。
- `tokenizer_k/`
  - 已训练好的 tokenizer 配置和权重，供预训练、SFT 与推理直接加载。
- `pretrain_code/`
  - `model.py`：从零实现的 Decoder-only Transformer（支持 RoPE、RMSNorm、权重共享、Flash Attention），封装为 `Transformer` 与 `ModelConfig`。
  - `pretrain.py`：使用自建 `PretrainDataset` 在本地语料上进行从头预训练，带学习率余弦退火、梯度累积、混合精度训练与 SwanLab 日志记录。
  - `pretrain_ds.py`：基于 HuggingFace `Trainer` + DeepSpeed 的预训练脚本示例，可从 Qwen 等现有模型继续预训练。
- `SFT_code/`
  - `model.py`：与预训练相同的 Transformer 结构，用于加载预训练权重后进行下游微调。
  - `sft.py`：在监督微调数据上进行 SFT 训练（如 `BelleGroup_sft.jsonl`），支持多卡、混合精度、梯度裁剪与 SwanLab 监控，从 `../model/pretrain_1024_18_6144.pth` 加载 base 模型。
  - `sft_ds.py`：使用 HuggingFace `Trainer` 的 SFT 管线，实现指令/多轮对话数据的模板化、mask 构造与训练。
- 根目录下 Python 脚本
  - `dataset.py`：定义 `PretrainDataset` 与 `SFTDataset`，负责从 `jsonl` 流式读取数据、拼接 BOS/EOS、padding 以及构造自定义 `loss_mask`。
  - `model_sample.py`：简单的推理与采样 demo，封装为 `TextGenerator` 类，可加载预训练或 SFT 检查点，给定 prompt 生成多条回复。
- `data/`
  - `mobvoi_seq_monkey_general_open_corpus.jsonl`：预训练语料示例。
  - `BelleGroup_sft.jsonl`、`seq_monkey_datawhale.jsonl` 等：SFT/预训练用指令或通用文本数据示例。
- `note/`
  - 若干 `*.ipynb` 与 `*.md`（如 `LLM.md`、`PretrainLM.md`、`Transformer.md`），记录理论推导、实验过程与实践笔记。
- `model/` 与 `swanlog/`
  - 存放训练好的模型权重（如 `happy-llm-215M-base`、预训练 checkpoint）和 SwanLab 运行日志、配置等。

## 功能模块说明

- **Tokenizer 训练**
  - 使用 `tokenizers` 库构建 BPE 模型，支持 ByteLevel 预分词与特殊 token：
    - `<|im_start|>`、`<|im_end|>` 等对话边界
    - 内置 `chat_template`，与 Qwen 等开源模型风格兼容。
- **从零实现 Transformer 预训练**
  - 纯 PyTorch 实现 Decoder-only Transformer：
    - 多头注意力 + RoPE 位置编码
    - RMSNorm + MLP（SiLU 激活）
    - 词嵌入与输出层权重共享
  - 训练脚本：
    - 支持余弦学习率调度、梯度累积、混合精度（`torch.amp`）、多 GPU `DataParallel`。
    - 使用 SwanLab 记录 loss、lr 等指标。
- **SFT（监督微调）**
  - 支持两种路线：
    - 基于自定义 Transformer + tokenizer 的 SFT（`SFT_code/sft.py` + `SFTDataset`）。
    - 基于 HuggingFace `Trainer` + 现成大模型（如 Qwen）的 SFT（`SFT_code/sft_ds.py`）。
  - 针对对话格式数据，构造精细的 `loss_mask`，只对 assistant 段落计算 loss。
- **HuggingFace/DeepSpeed 管线示例**
  - `pretrain_code/pretrain_ds.py` 展示如何：
    - 使用 `datasets` + `AutoTokenizer` 对大规模语料进行切块（block packing）。
    - 配合 `Trainer`、DeepSpeed 配置实现高效预训练。
- **推理与采样**
  - `model_sample.py` 提供统一的 `TextGenerator`：
    - 从 checkpoint 加载预训练或 SFT 模型。
    - 支持 temperature、top-k 采样、批量生成。

## 快速开始

### 环境依赖（简要）

建议使用 Python 3.10+，核心依赖包括（不完全列表）：

- `torch`（含 CUDA 支持）
- `transformers`
- `tokenizers`
- `datasets`
- `deepspeed`
- `torchdata`
- `swanlab`

可以根据实际环境使用 `pip install -r requirements.txt`（可自行整理）或手动安装以上依赖。

### 1. 训练自定义 Tokenizer

在 `tokenizer_code/` 下执行：

```bash
cd tokenizer_code
python train_tokenizer.py
```

默认会从 `../data/mobvoi_seq_monkey_general_open_corpus.jsonl` 读取语料，并在 `../tokenizer_k` 目录下生成 tokenizer。

### 2. 从零预训练 Tiny LLM

使用自研 Transformer + 自定义 tokenizer 进行预训练：

```bash
cd pretrain_code
python pretrain.py \
  --data_path ../data/seq_monkey_datawhale.jsonl \
  --out_dir ../model \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 2e-4 \
  --device cuda:0
```

训练完成后，会在 `out_dir` 下生成 `pretrain_1024_18_6144.pth` 等 checkpoint。

### 3. 使用指令数据进行 SFT

在预训练权重基础上继续做监督微调：

```bash
cd SFT_code
python sft.py \
  --data_path ../data/BelleGroup_sft.jsonl \
  --out_dir ../sft_model_215M \
  --epochs 1 \
  --batch_size 64 \
  --learning_rate 2e-4
```

或使用 HuggingFace `Trainer` 版本（需要准备好 `TrainingArguments` 等命令行参数）：

```bash
python sft_ds.py \
  --model_name_or_path <预训练模型路径> \
  --train_files ../data/BelleGroup_sft.jsonl \
  --output_dir ../sft_output
```

### 4. 本地推理示例

在根目录使用 `model_sample.py` 进行采样：

```bash
python model_sample.py
```

可以在脚本中修改：

- checkpoint 路径（预训练或 SFT 模型）
- 初始 prompt、采样温度、top-k 等参数

## 后续计划

- 补充 RLHF/RLAIF 等强化学习阶段代码与文档。
- 丰富中文/多轮对话任务的评测与案例。