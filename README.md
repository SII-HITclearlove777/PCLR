# PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning

<p align="center">
  <b>Weicheng Meng<sup>1,2</sup>, Jingyang Qiao<sup>2,3</sup>, Shaohui Liu<sup>1,2&dagger</sup>, Zhizhong Zhang<sup>3,4&dagger</sup>, Yuan Xie<sup>2,3&ddagger;</sup></b>
</p>

<p align="center">
  <sup>1</sup>Harbin Institute of Technology &nbsp;
  <sup>2</sup>Shanghai Innovation Institute &nbsp;
  <sup>3</sup>East China Normal University &nbsp;
  <sup>4</sup>Shanghai Key Laboratory of Computer Software Evaluating and Testing
</p>

Official PyTorch implementation for the ICLR 2026 paper **"PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning"**.

---

## Abstract

Continual Instruction Tuning (CIT) enables Large Multimodal Models (LMMs) to rapidly adapt to new tasks without retraining, but it suffers from the catastrophic forgetting problem. By adding new branches, model extension provides a great idea to accommodate novel knowledge while causing huge memory consumption. To jointly address forgetting and memory explosion, we propose the Compression-Integration-Learning (CIL) pipeline, which draws on the memory consolidation processes during human sleep. Compression streamlines old parameters to release capacity. Integration merges knowledge from similar tasks to restore the performance loss due to compression. For example, based on LLaVA-7B, the forgetting is reduced from 11.29 to 5.09. Learning reallocates released capacity for new task-relevant parameters. Next, based on the characteristics of LMMs at different learning stages, we establish the progressive learning process, further reducing forgetting from 5.09 to 3.39. Moreover, to adapt this process, we decompose LoRA into a set of rank vectors and introduce an extremely fine-grained architecture, LoRA Rank Pool (LRP), with the goal of flexible knowledge employment and editing. Finally, we combine all components, and yield Progressively Compressed LoRA (PCLR). Extensive experiments demonstrate that PCLR owns a memory budget close to non-extension methods while outperforming extension methods in performance.

## Highlights

- **Compression-Integration-Learning (CIL) Pipeline** -- Inspired by human memory consolidation during sleep, CIL compresses old parameters, integrates similar task knowledge, and reallocates capacity for new tasks.
- **LoRA Rank Pool (LRP)** -- Decomposes LoRA into individual rank vectors with a key-query selection mechanism, enabling extremely fine-grained knowledge employment and editing.
- **Progressive Learning** -- Adapts the learning strategy to different stages of CIT, further reducing catastrophic forgetting (from 5.09 to 3.39 on LLaVA-7B).
- **Near-zero Memory Overhead** -- Achieves memory budgets close to non-extension methods while outperforming extension-based approaches.

## Project Structure

```
PCLR/
├── lrp/                    # LRP on LLaVA-1.5 (Vicuna/LLaMA backbone)
│   ├── model/              #   LrpModel, LrpTSModel, LRP modules
│   ├── train/              #   Training entry points (train.py, train_mem.py)
│   ├── eval/               #   Evaluation scripts for 8 benchmarks
│   └── scripts/            #   Shell scripts for training & evaluation
├── lrp_llava_hf/           # LRP on LLaVA (HuggingFace native)
│   ├── model/              #   LrpModel, LrpTSModel adapted for HF LLaVA
│   ├── train/              #   Training entry points
│   ├── eval/               #   Level-based evaluation + BLEU/IoU metrics
│   └── scripts/            #   Shell scripts (15 benchmarks)
├── lrp_qwen/               # LRP on Qwen-VL
│   ├── model/              #   LrpModel, LrpTSModel adapted for Qwen-VL
│   ├── train/              #   Training entry points
│   └── scripts/            #   Shell scripts
├── QwenVL/                 # Qwen-VL model implementation
│   └── Qwen_model/         #   Configuration, modeling, tokenization, visual encoder
└── myllava/                # Custom LLaVA wrapper (LlavaLlamaForCausalLM)
    └── model/              #   LLaVA model & architecture definitions
```

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 1.13
- CUDA >= 11.7
- 4x GPUs (recommended: A100 80GB)

### Setup

```bash
git clone https://github.com/xxx/PCLR.git
cd PCLR
pip install -e .
```

### Required Pre-trained Models

Download and place the following in the project root:

| Model | Path | Source |
|-------|------|--------|
| Vicuna-7B-v1.5 | `./vicuna-7b-v1.5` | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| CLIP ViT-L/14@336 | `./clip-vit-large-patch14-336` | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) |
| MM Projector | `./vicuna-7b-v1.5/mm_projector.bin` | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |

## Data Preparation

Prepare the CoIN benchmark data following the [CoIN](https://github.com/zackschen/CoIN) repository. Update the paths in training/evaluation scripts:

- `--data_path`: Path to training JSON file (e.g., `/path/coin/json/Instructions-Origin/ScienceQA/train.json`)
- `--image_folder`: Path to image directory (e.g., `/path/coin/images`)

## Training

PCLR follows a sequential Continual Instruction Tuning process. Each task has two stages:

1. **Learning Stage** (`*_1.sh`) -- Learn new task knowledge using trainable rank pools
2. **Compression Stage** (`*_2.sh`) -- Compress old knowledge via Teacher-Student distillation (`--Teacher_Student True`)

### LLaVA-1.5 (8 tasks)

Run all tasks sequentially:

```bash
bash ./lrp/scripts/train/Train_all.sh
```

Or run individual tasks:

```bash
# Task 1: ScienceQA (Learning only)
bash ./lrp/scripts/train/1_ScienceQA_1.sh

# Task 4: GQA (Learning + Compression + Integration)
bash ./lrp/scripts/train/4_GQA_1.sh   # Learning
bash ./lrp/scripts/train/4_GQA_2.sh   # Compression
```

### LLaVA-HF (15 tasks)

```bash
bash ./lrp_llava_hf/scripts/train/Train_all.sh
```

### Qwen-VL (8 tasks)

```bash
bash ./lrp_qwen/scripts/train/Train_all.sh
```

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `--llm_train_rank_size` | Number of trainable rank vectors |
| `--llm_static_rank_size` | Number of static (frozen) rank vectors from prior tasks |
| `--llm_share_rank_size` | Number of shared rank vectors across tasks |
| `--llm_top_rank` | Top-k rank vectors selected via key-query mechanism |
| `--task_num` | Current task index (0-based) |
| `--Teacher_Student` | Enable Teacher-Student compression mode |
| `--lrp_model_path` | Path to previous checkpoint (for compression stage) |
| `--loss_weight3` | Weight for query alignment loss |
| `--skip_interval` | Compression skip interval |

## Evaluation

### LLaVA-1.5

Evaluate all benchmarks after the final task:

```bash
bash ./lrp/scripts/eval/Eval_all.sh
```

Evaluate a specific benchmark:

```bash
# Usage: bash <eval_script> <stage_name> <checkpoint_path>
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./lrp/scripts/eval/1_eval_sqa.sh ScienceQA-7b /path/to/checkpoint
```

### Supported Benchmarks

| # | Benchmark | Script | Metric |
|---|-----------|--------|--------|
| 1 | ScienceQA | `1_eval_sqa.sh` | Accuracy |
| 2 | TextVQA | `2_eval_textqa.sh` | Accuracy |
| 3 | ImageNet | `3_eval_ImageNet.sh` | Accuracy |
| 4 | GQA | `4_eval_gqa.sh` | Accuracy |
| 5 | VizWiz | `5_eval_vizwiz.sh` | Accuracy |
| 6 | Grounding | `6_eval_grounding.sh` | IoU |
| 7 | VQAv2 | `7_eval_vqav2.sh` | Accuracy |
| 8 | OCR-VQA | `8_eval_ocrvqa.sh` | Accuracy |

The LLaVA-HF variant additionally supports: ArxivQA, GeoChat, IconQA, ClevrMath, CodeQA, Flickr30k, DocVQA, MathQA, ChartQA, PathVQA, WikiQA.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{meng2026pclr,
  title={PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning},
  author={Meng, Weicheng and Qiao, Jingyang and Liu, Shaohui and Zhang, Zhizhong and Xie, Yuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgements

This project builds upon the following excellent works:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [CoIN](https://github.com/zackschen/CoIN)
- [Continual-NExT](https://github.com/ECNU-SII/Continual-NExT)
- [LoRA](https://github.com/microsoft/LoRA)

## License

This project is released under the [Apache 2.0 License](LICENSE).
