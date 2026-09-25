#!/usr/bin/env python3
"""
T18 bake-off: Nellie (Lefebvre et al., Nat Methods 2025), zero-shot, on the
unmasked slab image of every test cell (allen_export_test.py). Only Nellie's
filtering (multiscale Frangi) and labelling steps are used; the union of its
instance labels is the mask, written as <out>/<CellId>.png (0/255).

Runs in the Nellie venv (it has its own dependencies):
    /var/tmp/nba055/venvs/nellie/bin/python experiments/archive_demo/allen_nellie.py
"""
import argparse
import contextlib
import io
import logging
import os
import shutil
import tempfile

import numpy as np

TEST = os.environ.get('ALLEN_TEST', '/mnt/nas1/nba055-2/idea_1/archive_demo/allen/testset')
UM = 0.108333


def one(args):
    cid, img_dir, out, um, otsu, min_r = args
    import cv2
    import tifffile
    from nellie.im_info.verifier import FileInfo, ImInfo
    from nellie.segmentation.filtering import Filter
    from nellie.segmentation.labelling import Label
    logging.disable(logging.CRITICAL)
    import time
    im = cv2.imread(os.path.join(img_dir, f'{cid}.png'), cv2.IMREAD_GRAYSCALE)
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(dir=os.environ.get('TMPDIR', '/var/tmp'))
    try:
        p = os.path.join(tmp, f'{cid}.tif')
        tifffile.imwrite(p, im)
        with contextlib.redirect_stdout(io.StringIO()):
            fi = FileInfo(p, output_dir=os.path.join(tmp, 'out'))
            fi.find_metadata()
            fi.load_metadata()
            fi.change_axes('YX')
            fi.change_dim_res('X', um)
            fi.change_dim_res('Y', um)
            ii = ImInfo(fi)
            Filter(ii, remove_edges=False, device='cpu').run()
            Label(ii, device='cpu', otsu_thresh_intensity=otsu, min_radius_um=min_r).run()
            lab = np.asarray(ii.get_memmap(ii.pipeline_paths['im_instance_label']))
        dt = time.perf_counter() - t0
        cv2.imwrite(os.path.join(out, f'{cid}.png'), ((lab.reshape(im.shape) > 0) * 255).astype(np.uint8))
        return cid, '', dt
    except Exception as ex:
        return cid, f'{type(ex).__name__}: {ex}', np.nan
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nas1/nba055-2/idea_1/archive_demo/allen/preds/nellie')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--img-dir', default=None, help='default: <ALLEN_TEST>/img_raw')
    ap.add_argument('--um', type=float, default=UM, help='pixel size (um)')
    ap.add_argument('--timing', default=None, help='append per-image seconds to this CSV (dataset label = --label)')
    ap.add_argument('--label', default='ALLEN')
    ap.add_argument('--otsu', action='store_true', help="Nellie's otsu_thresh_intensity")
    ap.add_argument('--min-radius-um', type=float, default=0.25, help="Nellie's min_radius_um (default 0.25)")
    ap.add_argument('--ids', default=None, help='comma list of image stems (default: all in --img-dir)')
    a = ap.parse_args()
    img_dir = a.img_dir or f'{TEST}/img_raw'
    os.makedirs(a.out, exist_ok=True)
    ids = a.ids.split(',') if a.ids else sorted(p[:-4] for p in os.listdir(img_dir) if p.endswith('.png'))
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(one, [(c, img_dir, a.out, a.um, a.otsu, a.min_radius_um) for c in ids])
    if a.timing:
        new = not os.path.exists(a.timing)
        with open(a.timing, 'a') as f:
            if new:
                f.write('dataset,id,seconds\n')
            for c, e, t in res:
                f.write(f'{a.label},{c},{t}\n')
    errs = [f'{c}: {e}' for c, e, _ in res if e]
    print(f'{len(ids) - len(errs)}/{len(ids)} cells segmented; errors: {len(errs)}')
    for e in errs[:10]:
        print(' ', e)


if __name__ == '__main__':
    main()
