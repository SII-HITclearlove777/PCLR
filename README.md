# **PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning**

**Weicheng Meng,** **Jingyang Qiao,** **Zhizhong Zhang,** **Shaohui Liu†,** **Yuan Xie†**

1. Harbin Institute of Technology 
2. Shanghai Innovation Institute 
3. East China Normal University
4. Shanghai Key Laboratory of Computer Software Evaluating and Testing

Official Pytorch implementation for ICLR 2026 paper "**PCLR: Progressively Compressed LoRA for Multimodal Continual Instruction Tuning**"

## Abstract

Continual Instruction Tuning (CIT) enables Large Multimodal Models (LMMs) to rapidly adapt to new tasks without retraining, but it suffers from the catastrophic forgetting problem. By adding new branches, model extension provides a great idea to accommodate novel knowledge while causing huge memory consumption. To jointly address forgetting and memory explosion, we propose the Compression–Integration–Learning (CIL) pipeline, which draws on the memory consolidation processes during human sleep. Compression streamlines old parameters to release capacity. Integration merges knowledge from similar tasks to restore the performance loss due to compression. For example, based on LLaVA-7B, the forgetting is reduced from 11.29 to 5.09. Learning reallocates released capacity for new task-relevant parameters. Next, based on the characteristics of LMMs at different learning stages, we establish the progressive learning process, further reducing forgetting from 5.09 to 3.39. Moreover, to adapt this process, we decompose LoRA into a set of rank vectors and introduce an extremely fine-grained architecture, LoRA Rank Pool (LRP), with the goal of flexible knowledge employment and editing. Finally, we combine all components, and yield Progressively Compressed LoRA (PCLR). Extensive experiments demonstrate that PCLR owns a memory budget close to non-extension methods while outperforming extension methods in performance.
