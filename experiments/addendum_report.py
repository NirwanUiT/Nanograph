#!/usr/bin/env python3
"""Addendum A1/A2/A4 reporting helpers over results/paper/*/metrics.csv."""
import csv
import math
import os
import sys
from math import erf, sqrt


def _rows(name):
    p = os.path.join('results/paper', name, 'metrics.csv')
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def _has(x, k):
    return x.get(k) not in (None, '', 'nan')


def _f(x, k):
    return float(x[k])


def _sign_p(wins, losses):
    nn = wins + losses
    if nn == 0:
        return float('nan')
    z = (wins - nn / 2) / sqrt(nn / 4)
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def a1_paired(name):
    r = _rows(name)
    if r is None:
        print(f'--- {name}: (not finished) ---')
        return
    both = [x for x in r if _has(x, 'gt_iou') and _has(x, 'jpeg_gt_iou')]
    nofit = [x for x in r if _has(x, 'gt_iou') and not _has(x, 'jpeg_gt_iou')]
    d = [_f(x, 'gt_iou') - _f(x, 'jpeg_gt_iou') for x in both]
    wins = sum(1 for v in d if v > 0)
    loss = sum(1 for v in d if v < 0)
    ties = sum(1 for v in d if v == 0)
    m = sum(d) / len(d) if d else float('nan')
    mg = sum(_f(x, 'gt_iou') for x in both) / len(both)
    mj = sum(_f(x, 'jpeg_gt_iou') for x in both) / len(both)
    print(f'--- A1 {name}: n_total={len(r)} paired={len(both)} jpeg_no_fit={len(nofit)} ---')
    print(f'  mean gt_iou(paired)={mg:.4f}  mean jpeg_gt_iou(paired)={mj:.4f}')
    print(f'  mean(gt_iou-jpeg_gt_iou)={m:+.4f}  wins={wins} losses={loss} ties={ties}'
          f'  sign-test p={_sign_p(wins, loss):.3e}')


def a2_t3cost(name='org_default'):
    r = _rows(name)
    if r is None:
        print(f'--- A2 {name}: (not finished) ---')
        return
    pairs = [('pre_gt_iou', 'gt_iou'), ('pre_ng_fg_psnr', 'ng_fg_psnr'),
             ('pre_ng_psnr', 'ng_psnr'), ('pre_ng_ssim', 'ng_ssim')]
    print(f'--- A2 {name}: pre-compression minus decoded (n={len(r)}) ---')
    hdr = list(r[0].keys())
    for pre, post in pairs:
        if pre not in hdr or post not in hdr:
            print(f'  {pre} - {post}: MISSING ({pre in hdr=}, {post in hdr=})')
            continue
        dd = [_f(x, pre) - _f(x, post) for x in r if _has(x, pre) and _has(x, post)]
        if not dd:
            print(f'  {pre} - {post}: no data')
            continue
        mean = sum(dd) / len(dd)
        mx = max(dd, key=abs)
        print(f'  {pre:16s} - {post:12s}: mean={mean:+.5f}  max|Δ|={mx:+.5f}')


def a4_extras(name):
    r = _rows(name)
    if r is None:
        return
    def mean(k):
        v = [_f(x, k) for x in r if _has(x, k)]
        return sum(v) / len(v) if v else float('nan')
    def med(k):
        v = sorted(_f(x, k) for x in r if _has(x, k))
        return v[len(v) // 2] if v else float('nan')
    keys = ['seg_precision', 'seg_recall', 'ng_bytes', 'gt_cycle_rank',
            'recon_cycle_rank', 'graph_n_cycles', 'ng_mean_width', 'mean_width']
    print(f'--- A4 {name} ---')
    hdr = list(r[0].keys())
    for k in keys:
        if k in hdr:
            print(f'  {k:18s} mean={mean(k):.4f}')
    brr = mean('ng_bytes') / 65536.0
    print(f'  BRR (ng_bytes/65536) = {brr:.5f}   (Claim 8 ref 0.0365)')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = ['org_default', 'org_classical', 'org_replace']
    if which in ('a1', 'all'):
        for n in names:
            a1_paired(n)
    if which in ('a2', 'all'):
        a2_t3cost('org_default')
    if which in ('a4', 'all'):
        for n in names:
            a4_extras(n)
