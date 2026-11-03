# LoRA 复现：Low-Rank Adaptation of Large Language Models

对论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
（Hu et al., ICLR 2022）的独立复现。实现包含 RoBERTa-base 上的 GLUE 八个任务、
GPT-2 medium 上的 E2E NLG，以及秩/适配位置消融、子空间分析和显存/时延测量。

完整实验报告：[`report/report.pdf`](report/report.pdf)

## 主要结果

**GLUE（RoBERTa-base，r=8，α=8，单随机种子）**

| 任务 | 论文 LoRA | 复现 LoRA | 论文 FT | 复现 FT |
|---|---:|---:|---:|---:|
| MNLI | 87.5 | 86.4 | 87.6 | — |
| SST-2 | 95.1 | 95.0 | 94.8 | 94.6 |
| MRPC | 89.7 | 88.0 | 90.2 | 90.2 |
| CoLA | 63.4 | 61.6 | 63.6 | 62.2 |
| QNLI | 93.3 | 92.4 | 92.8 | — |
| QQP | 90.8 | 89.0 | 91.9 | — |
| RTE | 86.6 | 77.6 | 78.7 | 81.2 |
| STS-B | 91.5 | 91.2 | 91.2 | — |
| **平均** | **87.2** | **85.1** | **81.8** | **82.1** |

RTE 的差距来自初始化：按论文改用 MNLI 上的 LoRA 检查点初始化后，RTE/MRPC/STS-B
分别达到 86.6/89.2/91.6（论文 86.6/89.7/91.5）。

**E2E NLG（GPT-2 medium，r=4，α=32）**

| 方法 | 可训练参数 | BLEU | METEOR* | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|
| 全量微调 | 354.8M | 65.0 | 73.0 | 70.1 | 1.72 |
| LoRA | 0.39M | 67.1 | 73.7 | 70.3 | 1.83 |

\* METEOR 使用 NLTK 实现，与论文的 Java METEOR 不可直接比较；BLEU/ROUGE-L/CIDEr
均按 MR 聚合多参考后计算。

## 目录结构

```
src/       模型实现、训练/评测脚本
report/    LaTeX 源码、编译好的 PDF、图与自动生成的表
results/   每次运行的 result.json、预测、指标与 LoRA 权重
logs/      完整训练日志
```

不包含官方仓库克隆、E2E 原始数据和论文 PDF（见下文“准备”）；`results/` 中的
`.pt` 权重也未提交。

## 环境

- Python 3.12，依赖见 [`requirements.txt`](requirements.txt)
- GLUE 通过 HuggingFace `datasets` 在线加载；E2E 需要本地 CSV

## 准备

```bash
pip install -r requirements.txt

# 官方实现，仅用于核对超参数与检查点（可选）
git clone https://github.com/microsoft/LoRA official/LoRA

# E2E NLG 数据（trainset.csv / devset.csv / testset_w_refs.csv）
git clone https://github.com/tuetschek/e2e-dataset data/e2e
```


## 复现步骤

```bash
# GLUE 主实验示例（完整命令见报告附录 A）
python src/train_glue.py --task sst2 --method lora --rank 8 --alpha 8 \
  --epochs 10 --batch_size 16 --lr 5e-4 --max_len 256
python src/train_glue.py --task rte --method lora --rank 8 --alpha 16 \
  --epochs 20 --batch_size 32 --lr 5e-4 --max_len 256 --tag initoff_rte \
  --init_lora official/LoRA/examples/NLU/roberta_base_lora_mnli.bin

# E2E NLG
python src/train_e2e.py --method lora --rank 4 --alpha 32 --epochs 5 \
  --batch_size 8 --lr 2e-4 --tag e2e_lora_r4
python src/eval_e2e.py --run results/e2e/e2e_lora_r4

# 分析与作图
python src/efficiency.py
python src/subspace.py --ckpt_a results/glue/abl_rank8_sst2/best_lora.pt --rank_a 8 \
  --ckpt_b results/glue/abl_rank64_sst2/best_lora.pt --rank_b 64 --layer 6
python src/make_tables.py && python src/make_figures.py
```


## 实现说明

- `lora_lib.py` 实现了 `nn.Linear` 与 GPT-2 `Conv1D` 两种适配层，支持注入、冻结、
  合并/解合并；初始化时 B=0，因此训练起点与预训练模型严格等价。
- GLUE 与 E2E 的数据、超参、评测均按论文与官方仓库核对；E2E 测试集按 MR 聚合，
  平均每个 MR 有 7.4 条参考。
- 与论文的差异（训练轮数、单种子、METEOR 实现、生成超参等）在报告第 9 节逐条列出。
