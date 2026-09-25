#!/usr/bin/env python3
"""
Direction checks: every manuscript sentence whose wording depends on the
direction of a result, as an explicit test on the macros in numbers.tex.

Each check prints PASS / FAIL / TBD and the manuscript lines that use its
macros, so after a re-run only the FAIL lines need re-reading. Writes
<out> (markdown). Exit status 1 if any check FAILS.

Usage (from repo root):
    python paper/check_directions.py --numbers paper/numbers.tex \
        --tex paper/nanograph_main.tex --out results/paper/DIRECTIONS.md
"""
import argparse
import re
import sys

D = ['NComponents', 'TotalLength', 'MeanWidth', 'NBranches', 'NJunctions', 'CycleRank']
SETS = ['Uit', 'Cbmi', 'Mito', 'Human']


def num(v):
    v = v.replace('{,}', '').replace('\\pm', ' ').replace('$', '').strip()
    m = re.match(r'^([+-]?[0-9]*\.?[0-9]+)', v)
    return float(m.group(1)) if m else None


def frac(v):
    m = re.match(r'^(\d+)/(\d+)$', v.strip())
    return int(m.group(1)) / int(m.group(2)) if m else None


def ci(v):
    m = re.match(r'^\[([+-]?[0-9.]+),([+-]?[0-9.]+)\]$', v.strip())
    return (float(m.group(1)), float(m.group(2))) if m else None


def p_small(v, alpha=0.05):
    """make_numbers.fmt_p: '0.03' or '2.1\\times10^{-5}'."""
    m = re.match(r'^([0-9.]+)\\times10\^\{(-?\d+)\}$', v.strip())
    x = float(m.group(1)) * 10 ** int(m.group(2)) if m else num(v)
    return x is not None and x < alpha


CHECKS = [
    # (id, claim, macros, predicate(get) -> bool)
    ('layer-size', 'structure layer smaller than the lossless mask (abstract, layers section)',
     ['StructVsMaskX'], lambda g: num(g('StructVsMaskX')) > 1),
    ('layer-topology', 'stored components match the mask on >= 99 % and cycles on >= 90 % of images',
     ['GraphCompEqPct', 'GraphCycEqPct'],
     lambda g: num(g('GraphCompEqPct')) >= 99 and num(g('GraphCycEqPct')) >= 90),
    ('ds-every', 'stored graph closer to the reference than JPEG on every descriptor (tab:downstream)',
     [f'DsPWins{d}' for d in D] + [f'DsPWinsP{d}' for d in D],
     lambda g: all(frac(g(f'DsPWins{d}')) > frac(g(f'DsPLosses{d}')) and p_small(g(f'DsPWinsP{d}')) for d in D)),
    ('ds-components', 'JPEG route fragments components more than the stored graph',
     ['DsPJpegNComponentsBiasPct', 'DsPGraphNComponentsBiasPct', 'DsPJpegNComponentsCCC', 'DsPGraphNComponentsCCC'],
     lambda g: abs(num(g('DsPJpegNComponentsBiasPct'))) > abs(num(g('DsPGraphNComponentsBiasPct')))
     and num(g('DsPGraphNComponentsCCC')) > num(g('DsPJpegNComponentsCCC'))),
    ('ds-wasserstein', 'branch-length distribution closer to the reference than JPEG',
     ['DsPWassGraph', 'DsPWassJpeg'], lambda g: num(g('DsPWassGraph')) < num(g('DsPWassJpeg'))),
    ('cost', 'query cost: structure layer < mask re-analysis < JPEG route',
     ['DsTimeGraph', 'DsTimeRef', 'DsTimeJpeg'],
     lambda g: num(g('DsTimeGraph')) < num(g('DsTimeRef')) < num(g('DsTimeJpeg'))),
    ('gtiou-parity', 'GT-IoU at parity with JPEG (95 % CI of the mean difference contains 0)',
     ['DGTIoUWinsCI'], lambda g: ci(g('DGTIoUWinsCI'))[0] <= 0 <= ci(g('DGTIoUWinsCI'))[1]),
    ('gtiou-classical', 'with the classical cascade JPEG is ahead on GT-IoU',
     ['CGTIoUWinsPct'], lambda g: num(g('CGTIoUWinsPct')) < 50),
    ('truth-jpeg-length', 'JPEG length bias smaller than the graph\'s, but its concordance lower',
     ['TruthJpegLenBias', 'TruthGraphLenBias', 'TruthJpegLenCCC', 'TruthGraphLenCCC'],
     lambda g: abs(num(g('TruthJpegLenBias'))) < abs(num(g('TruthGraphLenBias')))
     and num(g('TruthJpegLenCCC')) < num(g('TruthGraphLenCCC'))),
    ('truth-width', 'every route overstates width, the simulator mask included',
     ['TruthRefWidthBias', 'TruthGraphWidthBias', 'TruthJpegWidthBias'],
     lambda g: all(num(g(f'Truth{a}WidthBias')) > 0 for a in ('Ref', 'Graph', 'Jpeg'))),
    ('truth-calibrated-ref', 'after calibration the graph is as close to truth as the simulator mask (within 2 points)',
     [f'Cal{a}{d}' for a in ('Graph', 'Ref') for d in ('Comp', 'Len', 'Width', 'Br', 'Junc', 'Cyc')],
     lambda g: all(abs(num(g(f'CalGraph{d}')) - num(g(f'CalRef{d}'))) <= 2
                   for d in ('Comp', 'Len', 'Width', 'Br', 'Junc', 'Cyc'))),
    ('truth-calibrated-jpeg', 'after calibration the graph is closer to truth than JPEG on components, length, branches, junctions',
     [f'Cal{a}{d}' for a in ('Graph', 'Jpeg') for d in ('Comp', 'Len', 'Br', 'Junc')],
     lambda g: all(num(g(f'CalGraph{d}')) < num(g(f'CalJpeg{d}')) for d in ('Comp', 'Len', 'Br', 'Junc'))),
    ('clip-rule', 'the one-diameter rule reduces the clip\'s component overcount',
     ['ClipTruthCompZero', 'ClipTruthCompAuto'],
     lambda g: abs(num(g('ClipTruthCompAuto'))) < abs(num(g('ClipTruthCompZero')))),
    ('real-seg', 'real training raises in-pipeline Seg-IoU on all four datasets',
     [f'RealSegIoU{p}{s}' for p in ('Sim', 'Real') for s in SETS],
     lambda g: all(num(g(f'RealSegIoUReal{s}')) > num(g(f'RealSegIoUSim{s}')) for s in SETS)),
    ('real-human', 'on EP-UiT-Human real training shrinks the branch bias and raises its CCC',
     ['RealHumanBrBiasSim', 'RealHumanBrBiasReal', 'RealHumanBrCCCSim', 'RealHumanBrCCCReal'],
     lambda g: abs(num(g('RealHumanBrBiasReal'))) < abs(num(g('RealHumanBrBiasSim')))
     and num(g('RealHumanBrCCCReal')) > num(g('RealHumanBrCCCSim'))),
    ('real-q1', 'the smallest JPEG is markedly worse than the structure layer (UiT, Human)',
     ['RealJpegQoneBrCCCUit', 'RealBrCCCUit', 'RealJpegQoneBrCCCHuman', 'RealBrCCCHuman'],
     lambda g: all(num(g(f'RealJpegQoneBrCCC{s}')) < num(g(f'RealBrCCC{s}')) - 0.1 for s in ('Uit', 'Human'))),
    ('real-q20', 'JPEG q20 reaches the structure layer on CBMI and MITO but falls short on UiT and Human',
     [f'RealJpegQtwentyBrCCC{s}' for s in SETS] + [f'RealBrCCC{s}' for s in SETS],
     lambda g: all(num(g(f'RealJpegQtwentyBrCCC{s}')) >= num(g(f'RealBrCCC{s}')) - 0.02 for s in ('Cbmi', 'Mito'))
     and all(num(g(f'RealJpegQtwentyBrCCC{s}')) < num(g(f'RealBrCCC{s}')) for s in ('Uit', 'Human'))),
    ('lossless-smallest', 'T14: the structure layer is smaller than the best lossless alternative (per dataset)',
     [f'LLRatio{s}' for s in ['Organelle'] + SETS],
     lambda g: all(num(g(f'LLRatio{s}')) < 1 for s in ['Organelle'] + SETS)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numbers', default='paper/numbers.tex')
    ap.add_argument('--tex', default='paper/nanograph_main.tex')
    ap.add_argument('--out', default='results/paper/DIRECTIONS.md')
    a = ap.parse_args()
    M = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', open(a.numbers).read(), re.M))
    tex = open(a.tex).read().split('\n')
    lines, fails = ['# Direction checks', '', f'numbers: `{a.numbers}` (commit {M.get("RunCommit", "?")})', '',
                    '| check | status | claim | manuscript lines |', '|---|---|---|---|'], 0
    for cid, claim, macros, pred in CHECKS:
        used = sorted({i + 1 for i, ln in enumerate(tex) for m in macros if re.search(r'\\' + m + r'(?![A-Za-z])', ln)})
        vals = {m: M.get(m, '\\tbd') for m in macros}
        if any('tbd' in v for v in vals.values()):
            st = 'TBD'
        else:
            try:
                st = 'PASS' if pred(lambda k: vals[k]) else 'FAIL'
            except Exception as ex:
                st = f'ERROR ({type(ex).__name__})'
        fails += st != 'PASS' and st != 'TBD'
        lines.append(f'| `{cid}` | **{st}** | {claim} | {", ".join(map(str, used)) or "-"} |')
    lines += ['', '## Values', ''] + [f'- `{cid}`: ' + ', '.join(f'{m}={M.get(m, "?")}' for m in macros)
                                    for cid, _, macros, _ in CHECKS]
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:len(CHECKS) + 6]))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
