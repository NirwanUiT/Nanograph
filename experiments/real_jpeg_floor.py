import os, sys, numpy as np, pandas as pd, cv2, torch
sys.path.insert(0,'experiments'); sys.path.insert(0,'..')
import downstream_morphometry as dm
from nanograph_v4 import NanographConfig
from nanograph_v4.segment import learned_segment
from nanograph_v4.detect import detect_polarity
from nanograph_v4.unet_seg import load_unet
REAL='/mnt/nas1/nba055-2/idea_1/real_mito'
def work(chunk):
    torch.set_num_threads(1); cfg=NanographConfig().for_real_mito(); M=load_unet(cfg.segment.learned_ckpt,'cpu'); rows=[]
    for d,tid in chunk:
        img=cv2.imread(f'{REAL}/{d}/images/{tid}.png',0); gt=cv2.imread(f'{REAL}/{d}/masks/{tid}.png',0)
        dr=dm.descriptors_at(dm.pixel_arm_table(gt),'auto',min_len='auto')[0]
        for q in (1,5,10,20):
            _,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,q]); dec=cv2.imdecode(buf,0)
            x=255-dec if detect_polarity(dec,cfg=cfg) else dec
            dj=dm.descriptors_at(dm.pixel_arm_table(learned_segment(M,x,cfg=cfg,device='cpu')),'auto',min_len='auto')[0]
            rows.append({'dataset':d,'id':tid,'q':q,'bytes':len(buf),**{k:dj[k] for k in dm.DESCRIPTORS},**{k+'_ref':dr[k] for k in dm.DESCRIPTORS}})
    return rows
if __name__=='__main__':
    import multiprocessing as mp
    Mf=pd.read_csv(f'{REAL}/manifest.csv'); Mf=Mf[Mf.split=='test']; jobs=list(zip(Mf.dataset,Mf.id)); ch=[jobs[i::12] for i in range(12)]
    with mp.get_context('spawn').Pool(12) as p: rows=[r for rs in p.map(work,ch) for r in rs]
    P=pd.DataFrame(rows); P.to_csv('results/real/real_downstream/jpeg_low_quality.csv',index=False)
    G=pd.read_csv('results/real/real_downstream/per_tile.csv'); G=G[(G.arm=='GRAPH7')&(G.L=='auto')]
    for d in ['UIT','CBMI','MITO','HUMAN']:
        g=G[G.dataset==d]; out=[f"STRUCTURE {g.structure_bytes.mean():.0f} B: " + ', '.join(f"{k.split('_')[1] if k!='total_length_px' else 'length'} {dm.agreement(g[k].to_numpy(float), g[k].to_numpy(float)*0+1)['n']*0 if False else ''}" for k in [])]
        rr=P[P.dataset==d]
        print(f'\n{d}: structure layer {g.structure_bytes.mean():.0f} B (analysis = mask analysis exactly)')
        for q,h in rr.groupby('q'):
            s=[]
            for k in ['n_branches','n_junctions','total_length_px']:
                ag=dm.agreement(h[k].to_numpy(float),h[k+'_ref'].to_numpy(float)); s.append(f"{k.replace('n_','').replace('_px','')} bias {ag['bias_pct']:+.0f}% CCC {ag['ccc']:.2f}")
            print(f'  JPEG q{q:<3d} {h.bytes.mean():6.0f} B | '+' | '.join(s))
        s=[]
        for k in ['n_branches','n_junctions','total_length_px']:
            ag=dm.agreement(g[k].to_numpy(float),g[k+'_ref'].to_numpy(float)) if k+'_ref' in g else None
        R=pd.read_csv('results/real/real_downstream/per_tile.csv'); R=R[(R.L=='auto')&(R.dataset==d)]; W={a:x.set_index('id') for a,x in R.groupby('arm')}
        ids=W['REF'].index
        print('  GRAPH7 structure | '+' | '.join(f"{k.replace('n_','').replace('_px','')} bias {dm.agreement(W['GRAPH7'].loc[ids,k].to_numpy(float),W['REF'].loc[ids,k].to_numpy(float))['bias_pct']:+.0f}% CCC {dm.agreement(W['GRAPH7'].loc[ids,k].to_numpy(float),W['REF'].loc[ids,k].to_numpy(float))['ccc']:.2f}" for k in ['n_branches','n_junctions','total_length_px']))
