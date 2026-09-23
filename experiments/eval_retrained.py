"""
Evaluate the retrained multi-domain U-Net on held-out val splits and render
before/after sample panels.

Reports TWO inference numbers per dataset:
  as-is      : feed the image directly (no polarity detection)  <- honest deploy
  best-of-2  : keep the better of {as-is, inverted} using GT      <- upper bound

If as-is ~= best-of-2 the network is genuinely polarity-agnostic and needs no
inference-time polarity detector.
"""
import os
import sys
import numpy as np

# nanograph_v4 is a symlink in the repo parent dir (three levels up from experiments/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cv2
import torch

from nanograph_v4.unet_seg import UNet, unet_predict
from train_unet_multidomain import build_splits, _norm

_HERE = os.path.dirname(os.path.abspath(__file__))
# checkpoint may live in this experiments/ or in the sibling repo-parent experiments/
_CKPT_CANDIDATES = [
    os.path.join(_HERE, 'unet_multidomain.pt'),
    os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'experiments', 'unet_multidomain.pt'),
]
CKPT = next((p for p in _CKPT_CANDIDATES if os.path.exists(p)), _CKPT_CANDIDATES[0])
OUT = os.path.join(_HERE, 'retrained_eval')
csv_path = None


def iou_dice(pred, gt):
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    return float(iou), float(dice)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(CKPT, map_location=device)
    model = UNet(base=ck.get('base', 32)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f'loaded {CKPT} (epoch {ck["epoch"]})')

    splits = build_splits(seed=0)
    rows = []
    for ds in splits:
        asis, best = [], []
        panels = []
        for ip, mp in splits[ds]['val']:
            g = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            raw = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            img = (_norm(raw) * 255).astype(np.uint8)
            m0 = unet_predict(model, img, device=device)
            m1 = unet_predict(model, 255 - img, device=device)
            i0, d0 = iou_dice(m0, g)
            i1, d1 = iou_dice(m1, g)
            asis.append((i0, d0))
            best.append((i0, d0) if i0 >= i1 else (i1, d1))
            if len(panels) < 4:
                panels.append((raw, g, m0))
        ai = np.mean([a[0] for a in asis]); ad = np.mean([a[1] for a in asis])
        bi = np.mean([b[0] for b in best]); bd = np.mean([b[1] for b in best])
        rows.append((ds, ai, ad, bi, bd))
        print(f'{ds:13s} as-is IoU={ai:.3f} Dice={ad:.3f} | best-of-2 IoU={bi:.3f} Dice={bd:.3f}')

        # panel
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(panels)
        fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            ax = ax[None]
        for r, (raw, g, m) in enumerate(panels):
            ax[r, 0].imshow(raw, cmap='gray'); ax[r, 0].set_title('input')
            ax[r, 1].imshow(g, cmap='gray'); ax[r, 1].set_title('GT')
            ax[r, 2].imshow(m, cmap='gray'); ax[r, 2].set_title('retrained U-Net')
            for c in range(3):
                ax[r, c].axis('off')
        plt.suptitle(ds)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'{ds}_panel.png'), dpi=110)
        plt.close()

    print('\nsummary (held-out val):')
    print(f'{"dataset":13s} {"as-is IoU":>10s} {"best IoU":>10s}')
    for ds, ai, ad, bi, bd in rows:
        print(f'{ds:13s} {ai:>10.3f} {bi:>10.3f}')

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write('dataset,as_is,oracle\n')
            for ds, ai, ad, bi, bd in rows:
                f.write(f'{ds},{ai:.6f},{bi:.6f}\n')
        print(f'wrote {csv_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='write dataset,as_is,oracle CSV')
    args = ap.parse_args()
    csv_path = args.csv
    main()
