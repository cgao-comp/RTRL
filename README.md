# RTRL

The implementation for the paper "**Robust Temporal Residual Learning for Rare Anomaly Detection in Continuous-Time Dynamic Graphs**", WSDM 2027.

---

## Repository Structure

```
RTRL/
├── code/                        # all source code
│   ├── train_gan.py             # entry point: training + evaluation loop
│   ├── test.py                  # load a saved checkpoint and evaluate it
│   ├── option.py                # all hyper-parameters (argparse)
│   ├── paths.py                 # repo-root based directory resolution
│   ├── datasets.py              # DygDataset + collate_fn (temporal neighbour sampling)
│   ├── utils.py                 # NeighborFinder, logger
│   ├── process_data.py          # random-walk based training-set subsampling
│   ├── gen_data.sh              # run process_data.py on all 6 datasets
│   ├── run_all.sh               # run the main experiments on all 6 datasets
│   ├── model/
│   │   ├── wllm.py              # the whole model (train_forward / infer_forward)
│   │   ├── llm_wb.py            # LLM branch: prompt + cross-attention + frozen layers
│   │   ├── llm_prompt.py        # task prompt & per-dataset descriptions
│   │   ├── cross_attention.py   # temporal tokens -> LLM token space
│   │   ├── Qwen_simpled2.py     # standalone Qwen3 attention core (GQA + RoPE + RMSNorm)
│   │   ├── residual_his.py      # residual history + residual losses
│   │   ├── history_memory.py    # FIFO history bank
│   │   └── memory_net.py        # GRU global memory
│   └── modules/
│       ├── embedding_module.py  # stacked temporal attention
│       ├── temporal_attention.py# TGAT-style temporal attention layer
│       ├── time_encoding.py     # TGAT time encoder
│       └── utils.py             # feature MLPs / merge layers
├── data/                        # datasets, see Dataset section (not tracked by git)
│   ├── ml_<dataset>.csv         # edge list
│   ├── ml_<dataset>.npy         # edge features
│   ├── ml_<dataset>_node.npy    # node features
│   └── processed/               # output of process_data.py
├── saved_checkpoints/           # Qwen3 backbone and trained weights, see Dataset (not tracked by git)
├── train_log/                   # training logs
├── walk_log/                    # preprocessing logs
├── requirements.txt
└── README.md
```

---

## Installation

The code is tested with `Python 3.11.14`, `PyTorch 2.6.0+cu118`, `torch-scatter 2.1.2+pt26cu118`, `transformers 5.2.0` and `CUDA 11.8`.

```bash
conda create -n TLLM python=3.11 -y
conda activate TLLM
pip install -r requirements.txt
```

`requirements.txt` pins the versions listed below and already points pip at the CUDA 11.8 PyTorch index and the matching torch-scatter wheel index, so one command is enough. If you run a different CUDA / torch combination, install the two heavy packages from the matching indexes first and then the rest:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| Python | 3.11.14 |
| torch | 2.6.0+cu118 |
| torch-scatter | 2.1.2+pt26cu118 |
| transformers | 5.2.0 |
| numpy | 1.26.4 |
| pandas | 1.5.3 |
| scikit-learn | 1.6.1 |
| tqdm | 4.65.0 |
| CUDA | 11.8 |

`torch_scatter` is the only non-trivial dependency: it is used by the temporal attention (`scatter_softmax`, `scatter_add`) and by the global memory (`scatter`). It is a compiled extension, so it must match your exact torch / CUDA build — install it from the wheel index above rather than building from source.

The LLM backbone is **Qwen3-0.6B**, which is shipped in the release archive at `saved_checkpoints/LLM/Qwen3-0_6B` (see [Dataset](#dataset)). If that directory is empty, `model/llm_wb.py` downloads the weights from HuggingFace on first run and caches them there; to work offline, place the following files in it beforehand:

```
saved_checkpoints/LLM/Qwen3-0_6B/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── model.safetensors
```

---

## Dataset

We use six public CTDG datasets. The datasets, the preprocessed training splits and the Qwen3-0.6B backbone are shipped as one archive, because their combined size is far beyond what git can hold. Download it and extract it at the repository root so that `data/` and `saved_checkpoints/` are filled in:

> [Baidu Netdisk link](https://pan.baidu.com/s/1RS_vCQgn-AkJ0aB-w8BPBA?pwd=jcop)

```text
RTRL/
├── data/
│   ├── ml_<dataset>.csv
│   ├── ml_<dataset>.npy
│   ├── ml_<dataset>_node.npy
│   └── processed/ml_<dataset>_processed_train.csv
└── saved_checkpoints/
    └── LLM/Qwen3-0_6B/            Qwen3-0.6B config, tokenizer and weights
```

---

## Data Preprocessing

The anomaly ratio above is extremely low, and training on all edges of the 70% prefix is both slow and dominated by normal interactions. `process_data.py` therefore keeps a **random-walk-induced subgraph** around anomalous edges:

1. collect every node touched by a positive training edge as a seed;
2. build the adjacency list of the training prefix and run `num_walks` restarted random walks of length `walk_length` from each seed (`restart_prob = 0.15`);
3. keep the training edges whose two endpoints were both visited, and append the last `append_num` (default 1024) edges so that the most recent period is always complete.

```bash
bash code/gen_data.sh
# or a single dataset
cd code && python process_data.py --dataset_name wikipedia --walk_length 3 --num_walks 3
```

The result is written to `data/processed/ml_<name>_processed_train.csv` and logged to `walk_log/`. This step is **required** by default: `train_gan.py` reads the processed training set unless you pass `--use_processed_data` (which flips the flag off and trains on the raw prefix).

---

## Run Examples

Train and evaluate one dataset:

```bash
cd code
python train_gan.py --mode re_go_wiki --data_set wikipedia \
    --llm_model Qwen3-0_6B --llm_dim 1024 \
    --batch_size 256 --hidden_dim 256 --n_epochs 20 \
    --learning_rate_drop_step 7 --train_ratio 0.7 --device cuda:0
```

Reproduce the main table on all six datasets (pass the GPU id as the first argument):

```bash
bash code/run_all.sh 0
```

Training logs are written to `train_log/<mode>-<dataset>-<timestamp>.log`, and the checkpoint with the best test AUC is kept in `saved_checkpoints/train_weight/`.

---

## Loading and Validating a Checkpoint

`train_gan.py` stores `{"model_state_dict": ...}` in `saved_checkpoints/train_weight/` whenever the test AUC improves. `test.py` reloads such a file and scores it again, which is the quickest way to check that a saved model is intact and to report a number for it:

```bash
cd code
python test.py --data_set bitcoinalpha \
    --ckpt ../saved_checkpoints/train_weight/re_go_babitcoinalpha0-bitcoinalpha-0_xxxx-node-classification.pth \
    --batch_size 256 --hidden_dim 256 --device cuda:0
```

It prints how many tensors were restored, then AUC-ROC and AP together with the number of edges and positives for the requested split:

```text
checkpoint: ../saved_checkpoints/train_weight/re_go_babitcoinalpha0-bitcoinalpha-0_xxxx-node-classification.pth
  tensors restored: 86/86, missing: 0, unexpected: 0
test: samples=3618, positives=230, auc=<auc>, ap=<ap>
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chen2027robust,
  title     = {Robust Temporal Residual Learning for Rare Anomaly Detection in Continuous-Time Dynamic Graphs},
  author    = {Aoge Chen and Huixiang Zhang and Xianghua Li and Lianwei Wu and Chao Gao},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and Data Mining (WSDM '27)},
  year      = {2027}
}
```

---

## Acknowledgement

The temporal graph modules (`modules/temporal_attention.py`, `modules/time_encoding.py`, `utils.py`'s `NeighborFinder`) follow the design of [TGAT](https://arxiv.org/abs/2002.07962) and the [TGN](https://github.com/twitter-research/tgn) codebase. The LLM backbone is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). We thank the authors for releasing their code and weights.
