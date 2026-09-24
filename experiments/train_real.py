#!/usr/bin/env python3
"""
Train the organelle U-Net on real annotated mitochondria (T13).

Data: real_mito/manifest.csv (experiments/prep_real_mito.py); simulated data
(nmi_data) optional for joint training, restricted to the images outside
results/paper/heldout_organelle.txt.

Strategies
  scratch   random init, real data only
  finetune  init from a simulation-trained checkpoint (--init), real only,
            lower learning rate
  joint     random init, real + simulated tiles in every epoch

Model selection: best (val Dice + val centreline sensitivity) on the val
split of the training datasets. Test sets are never touched here.

Augmentation as experiments/train_unet.py (flips, rot90, gamma/scale/bias,
blur, noise), with per-worker RNG seeding (train_unet.py's loader workers
share one RNG state and so draw identical augmentations).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train_unet import combined_loss, evaluate  # noqa: E402
from train_unet_topo import skeleton_recall  # noqa: E402
from nanograph_v4.unet_seg import UNet  # noqa: E402

REAL = '/mnt/nas1/nba055-2/idea_1/real_mito'
NMI = '/mnt/nas1/nba055-2/idea_1/nmi_data'
HELDOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'paper', 'heldout_organelle.txt')


class PairSet(Dataset):
    """(image path, mask path) pairs -> augmented tensors (+ tubed skeleton)."""

    def __init__(self, pairs, train=True):
        self.pairs, self.train = pairs, train
        self.rng = np.random.default_rng(0)
        self.k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from skimage.morphology import skeletonize
        ip, mp = self.pairs[i]
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        msk = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        if self.train:
            r = self.rng
            if r.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if r.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = int(r.integers(0, 4))
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
            img = np.clip(img ** r.uniform(0.6, 1.6), 0, 1)
            img = np.clip(img * r.uniform(0.7, 1.3) + r.uniform(-0.1, 0.1), 0, 1)
            if r.random() < 0.5:
                img = cv2.GaussianBlur(img, (0, 0), float(r.uniform(0.5, 2.0)))
            if r.random() < 0.5:
                img = np.clip(img + r.normal(0, r.uniform(0.01, 0.06), img.shape).astype(np.float32), 0, 1)
        x, y = torch.from_numpy(img[None].copy()), torch.from_numpy(msk[None].copy())
        if not self.train:
            return x, y
        t = cv2.dilate(skeletonize(msk > 0.5).astype(np.uint8), self.k).astype(np.float32)
        return x, y, torch.from_numpy(t[None])


def _worker_init(wid):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def real_pairs(sets, split):
    M = pd.read_csv(os.path.join(REAL, 'manifest.csv'))
    M = M[M.dataset.isin(sets) & (M.split == split)]
    return [(os.path.join(REAL, d, 'images', i + '.png'), os.path.join(REAL, d, 'masks', i + '.png'))
            for d, i in zip(M.dataset, M.id)]


def sim_pairs():
    held = {s + '.png' for s in open(HELDOUT).read().split()}
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(NMI, 'org', '*.png')))
    return [(os.path.join(NMI, 'org', i), os.path.join(NMI, 'seg', i)) for i in ids if i not in held]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-sets', required=True, help='comma list of UIT,CBMI,MITO')
    ap.add_argument('--strategy', required=True, choices=['scratch', 'finetune', 'joint'])
    ap.add_argument('--init', default='', help='checkpoint for finetune')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skel-recall', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    sets = a.train_sets.split(',')
    tr = real_pairs(sets, 'train')
    if a.strategy == 'joint':
        tr = tr + sim_pairs()
    va = real_pairs(sets, 'val')
    lr = 3e-4 if a.strategy == 'finetune' else 1e-3
    print(f'device={device} sets={sets} strategy={a.strategy} train={len(tr)} val={len(va)} lr={lr}',
          flush=True)
    tl = DataLoader(PairSet(tr, True), batch_size=a.bs, shuffle=True, num_workers=4, drop_last=True,
                    worker_init_fn=_worker_init)
    vl = DataLoader(PairSet(va, False), batch_size=a.bs, shuffle=False, num_workers=2)
    model = UNet(base=32).to(device)
    if a.strategy == 'finetune':
        ck = torch.load(a.init, map_location=device)
        model.load_state_dict(ck['model'] if 'model' in ck else ck)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        losses = []
        for x, y, t in tl:
            x, y, t = x.to(device), y.to(device), t.to(device)
            opt.zero_grad()
            logit = model(x)
            loss = combined_loss(logit, y, cldice_w=0.5)
            if a.skel_recall > 0:
                loss = loss + a.skel_recall * skeleton_recall(logit, t).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        d, iou, cs = evaluate(model, vl, device)
        flag = ''
        if d + cs > best:
            best = d + cs
            torch.save({'model': model.state_dict(), 'base': 32, 'val_dice': d, 'val_iou': iou,
                        'val_csens': cs, 'epoch': ep, 'train_sets': sets, 'strategy': a.strategy,
                        'seed': a.seed}, a.out)
            flag = ' *saved'
        print(f'ep {ep:3d} loss={np.mean(losses):.4f} val_dice={d:.4f} val_iou={iou:.4f} '
              f'val_csens={cs:.4f}{flag}', flush=True)


if __name__ == '__main__':
    main()
