#!/usr/bin/env python3
"""
Organelle U-Net with a topology-aware objective, under a clean split.

Split: the 108 held-out images (results/paper/heldout_organelle.txt, the same
seed-0 split as train_unet.py) are never seen. Model selection uses a 10 %
validation carve-out of the remaining 618 images (seed 1), unlike
train_unet.py, which selected on the held-out images themselves.

Loss: train_unet.combined_loss (Dice + BCE + clDice), optionally plus the
skeleton-recall loss of Kirchhoff et al. (ECCV 2024):
    L_sr = 1 - sum(p * T) / sum(T),
T = the annotation's skeleton dilated by 1 px (a "tubed" skeleton), p the
predicted foreground probability. It penalises missed centreline, i.e.
broken or absent thin branches.

Usage (from repo root):
    python experiments/train_unet_topo.py --skel-recall 0.0 --out weights/unet_clean_base.pt
    python experiments/train_unet_topo.py --skel-recall 1.0 --out weights/unet_clean_skelrec.pt
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from train_unet import NMI, NMISeg, combined_loss, evaluate  # noqa: E402
from nanograph_v4.unet_seg import UNet  # noqa: E402

HELDOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'paper', 'heldout_organelle.txt')


class WithSkeleton(Dataset):
    """Adds the tubed skeleton of the (augmented) mask as a third tensor."""

    def __init__(self, base):
        self.base = base
        self.k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        m = y.numpy()[0] > 0.5
        t = cv2.dilate(skeletonize(m).astype(np.uint8), self.k)
        return x, y, torch.from_numpy(t.astype(np.float32))[None]


def skeleton_recall(logit, tube, eps=1.0):
    p = torch.sigmoid(logit)
    return 1 - ((p * tube).sum((1, 2, 3)) + eps) / (tube.sum((1, 2, 3)) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cldice', type=float, default=0.5)
    ap.add_argument('--skel-recall', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    held = {s + '.png' for s in open(HELDOUT).read().split()}
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(NMI, 'org', '*.png')))
    pool = [i for i in ids if i not in held]
    rng = np.random.default_rng(1)
    rng.shuffle(pool)
    n_val = int(0.1 * len(pool))
    val_ids, tr_ids = pool[:n_val], pool[n_val:]
    print(f'device={device} train={len(tr_ids)} val={len(val_ids)} test(held-out)={len(held)} '
          f'skel_recall={a.skel_recall}', flush=True)

    tr = DataLoader(WithSkeleton(NMISeg(tr_ids, train=True, seed=a.seed)), batch_size=a.bs,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(NMISeg(val_ids, train=False), batch_size=a.bs, shuffle=False, num_workers=2)
    model = UNet(base=a.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        losses = []
        for x, y, t in tr:
            x, y, t = x.to(device), y.to(device), t.to(device)
            opt.zero_grad()
            logit = model(x)
            loss = combined_loss(logit, y, cldice_w=a.cldice)
            if a.skel_recall > 0:
                loss = loss + a.skel_recall * skeleton_recall(logit, t).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        d, iou, cs = evaluate(model, va, device)
        score = d + cs                       # same selection rule as train_unet.py
        flag = ''
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'base': a.base, 'val_dice': d,
                        'val_iou': iou, 'val_csens': cs, 'epoch': ep,
                        'skel_recall': a.skel_recall, 'split': 'heldout-excluded, val seed 1'},
                       a.out)
            flag = ' *saved'
        print(f'ep {ep:3d} loss={np.mean(losses):.4f} val_dice={d:.4f} val_iou={iou:.4f} '
              f'val_csens={cs:.4f}{flag}', flush=True)


if __name__ == '__main__':
    main()
