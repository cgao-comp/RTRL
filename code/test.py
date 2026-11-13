import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from option import args
import datasets as dataset
from model.wllm import wllm


def load_model(cfg, ckpt_path):
    train_ds = dataset.DygDataset(cfg, 'train')
    cfg.n_nodes = train_ds.num_nodes

    model = wllm(cfg).to(cfg.device)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = state.get('model_state_dict', state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    print(f'checkpoint: {ckpt_path}')
    print(f'  tensors restored: {loaded}/{len(state)}, missing: {len(missing)}, unexpected: {len(unexpected)}')
    if missing:
        print(f'  missing keys: {sorted({".".join(k.split(".")[:2]) for k in missing})}')
    if unexpected:
        print(f'  unexpected keys: {sorted(unexpected)}')
    return model


def evaluate(cfg, model, split):
    ds = dataset.DygDataset(cfg, split)
    collate = dataset.Collate(cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_data_workers, collate_fn=collate.dyg_collate_fn)

    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model.infer_forward(batch, cfg.device)
            preds.append(out['logits'].sigmoid().float().cpu().numpy().flatten())
            labels.append(batch['labels'].numpy().flatten())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    print(f'{split}: samples={len(labels)}, positives={int(labels.sum())}, auc={auc:.4f}, ap={ap:.4f}')
    return auc, ap


if __name__ == '__main__':
    if not args.ckpt:
        raise SystemExit('--ckpt is required, e.g. --ckpt ../saved_checkpoints/train_weight/<name>.pth')

    cfg = args
    model = load_model(cfg, cfg.ckpt)
    evaluate(cfg, model, cfg.split)
