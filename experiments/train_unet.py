"""
Train the Nanograph U-Net foreground segmenter on NMI organelle data.

Independent training set (nmi_data/org + seg, 726 paired fluorescence images
with clean binary masks). The trained model is applied ZERO-SHOT to other
modalities, so augmentation deliberately spans intensity / blur / noise /
geometry to bridge the domain gap (e.g. noisy-real NMI -> smooth-sim Aaron).

Outputs:
  experiments/unet_ckpt.pt          best-val checkpoint (by Dice)
  experiments/unet_train.png        loss + val-Dice/IoU curves
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nanograph_v4.unet_seg import UNet

NMI = '/mnt/nas1/nba055-2/idea_1/nmi_data'


class NMISeg(Dataset):
    def __init__(self, ids, size=256, train=True, seed=0):
        self.ids = ids
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def _norm(self, img):
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    def __getitem__(self, i):
        fid = self.ids[i]
        img = cv2.imread(os.path.join(NMI, 'org', fid), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(NMI, 'seg', fid), cv2.IMREAD_GRAYSCALE)
        img = self._norm(img)
        msk = (msk > 0).astype(np.float32)

        if self.train:
            rng = self.rng
            # geometric: flips + 90-rot
            if rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if rng.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = int(rng.integers(0, 4))
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)
            # intensity: gamma + scale + bias
            img = np.clip(img ** rng.uniform(0.6, 1.6), 0, 1)
            img = np.clip(img * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1), 0, 1)
            # blur (bridge to smoother domains)
            if rng.random() < 0.5:
                s = float(rng.uniform(0.5, 2.0))
                img = cv2.GaussianBlur(img, (0, 0), s)
            # additive noise
            if rng.random() < 0.5:
                img = np.clip(img + rng.normal(0, rng.uniform(0.01, 0.06),
                                               img.shape).astype(np.float32), 0, 1)
        return (torch.from_numpy(img[None].copy()),
                torch.from_numpy(msk[None].copy()))


def dice_bce(logit, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + eps) / (union + eps)).mean()
    return bce + dice


# --- Differentiable soft-skeleton clDice (Shit et al., CVPR 2021) ---
# Centerline-aware loss: penalises thin-structure / connectivity errors that
# area-weighted Dice ignores (1-2px tips contribute ~0 to Dice). This is the
# direct fix for the U-Net under-tracing thin filaments.
def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skel(x, iters=10):
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(p, target, iters=10, smooth=1.0):
    """clDice loss on probabilities p and binary target, both (B,1,H,W)."""
    sk_p = soft_skel(p, iters)
    sk_t = soft_skel(target, iters)
    d = (1, 2, 3)
    tprec = (sk_p * target).sum(d).add(smooth) / sk_p.sum(d).add(smooth)
    tsens = (sk_t * p).sum(d).add(smooth) / sk_t.sum(d).add(smooth)
    cldice = 1 - 2.0 * (tprec * tsens) / (tprec + tsens)
    return cldice.mean()


def combined_loss(logit, target, cldice_w=0.5, iters=10):
    base = dice_bce(logit, target)
    if cldice_w <= 0:
        return base
    p = torch.sigmoid(logit)
    return base + cldice_w * soft_cldice(p, target, iters=iters)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious, csens = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prob = torch.sigmoid(model(x))
        p = (prob >= 0.5).float()
        inter = (p * y).sum(dim=(1, 2, 3))
        psum, ysum = p.sum(dim=(1, 2, 3)), y.sum(dim=(1, 2, 3))
        dices += ((2 * inter + 1) / (psum + ysum + 1)).cpu().tolist()
        ious += ((inter + 1) / (psum + ysum - inter + 1)).cpu().tolist()
        # centerline sensitivity: fraction of GT skeleton covered by prediction
        sk_t = soft_skel(y, 10)
        cs = ((sk_t * p).sum(dim=(1, 2, 3)) + 1) / (sk_t.sum(dim=(1, 2, 3)) + 1)
        csens += cs.cpu().tolist()
    return (float(np.mean(dices)), float(np.mean(ious)),
            float(np.mean(csens)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--base', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cldice', type=float, default=0.5,
                    help='weight of the centerline (clDice) loss term')
    ap.add_argument('--cldice-iters', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'unet_ckpt.pt'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(NMI, 'org', '*.png')))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_val = max(1, int(0.15 * len(ids)))
    val_ids, tr_ids = ids[:n_val], ids[n_val:]
    print(f'device={device} train={len(tr_ids)} val={len(val_ids)}')

    tr = DataLoader(NMISeg(tr_ids, train=True, seed=args.seed), batch_size=args.bs,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(NMISeg(val_ids, train=False), batch_size=args.bs,
                    shuffle=False, num_workers=2)

    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    hist = {'loss': [], 'dice': [], 'iou': []}
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        losses = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = combined_loss(model(x), y, cldice_w=args.cldice,
                                 iters=args.cldice_iters)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        d, iou, cs = evaluate(model, va, device)
        hist['loss'].append(float(np.mean(losses)))
        hist['dice'].append(d); hist['iou'].append(iou)
        # select for thin-structure recovery: Dice + centerline sensitivity
        score = d + cs
        flag = ''
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'base': args.base,
                        'val_dice': d, 'val_iou': iou, 'val_csens': cs,
                        'epoch': ep}, args.out)
            flag = ' *saved'
        print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  val_dice={d:.4f}  '
              f'val_iou={iou:.4f}  val_csens={cs:.4f}{flag}')

    print(f'\nbest val (dice+csens)={best:.4f} -> {args.out}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist['loss']); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(hist['dice'], label='val Dice')
    ax[1].plot(hist['iou'], label='val IoU')
    ax[1].set_title('validation'); ax[1].set_xlabel('epoch'); ax[1].legend()
    ax[1].grid(alpha=0.3)
    out_png = os.path.splitext(args.out)[0].replace('_ckpt', '_train') + '.png'
    plt.tight_layout(); plt.savefig(out_png, dpi=120)
    print(f'saved {out_png}')


if __name__ == '__main__':
    main()
