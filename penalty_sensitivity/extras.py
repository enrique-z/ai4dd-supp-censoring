import json, numpy as np, sys
sys.path.insert(0, '.')
from s4_optimizer_sensitivity import boot_targets
from scipy import stats
K=('rmse_naive','rmse_exact_only','rmse_censoring_aware','rmse_naive_matched')
IK={'rmse_naive':'info_naive','rmse_exact_only':'info_exact_only','rmse_censoring_aware':'info_censoring_aware','rmse_naive_matched':'info_naive_matched'}
raw=json.load(open('../evidence/censoring_raw.json'))['S4_downstream']['targets']
L={l:json.load(open(f's4_lam_{l}.json'))['targets'] for l in ('0.001','0.01','0.1','1')}
O=json.load(open('s4_orig_full.json'))['targets']
out={}
# 1 full regression gate
md=max(abs(a[k]-b[k]) for t,p in zip(O,raw) for a,b in zip(t['per_seed'],p['per_seed']) for k in K)
cf=sum(a[IK[k]]['converged']!=b[k.replace('rmse_','converged_')] for t,p in zip(O,raw) for a,b in zip(t['per_seed'],p['per_seed']) for k in K)
out['full_gate']={'max_abs_rmse_diff_all_158x10x4':md,'conv_flag_mismatches':int(cf)}
print('full gate',out['full_gate'])
# 2 corrected vs original at 1e-3: per-arm max per-target change
C=L['0.001']; nex=np.array([t['per_seed'][0]['n_train_exact_only'] for t in raw])
ch={}
for k in K:
    d=np.array([c[k]-o[k] for c,o in zip(C,O)])
    ch[k]={'mean':round(float(d.mean()),5),'max_abs':round(float(np.abs(d).max()),4),'n_targets_abs_gt_0.01':int((np.abs(d)>0.01).sum())}
out['corr_vs_orig_1e-3_per_arm']=ch; print(ch)
d=np.array([c['rmse_censoring_aware']-o['rmse_censoring_aware'] for c,o in zip(C,O)])
idx=np.argsort(d)[:8]
out['tobit_biggest_moves']=[{'tid':O[i]['tid'],'target':O[i]['target'][:45],'n_exact_train':int(nex[i]),'cens_frac':O[i]['censored_frac'],'tobit_orig':O[i]['rmse_censoring_aware'],'tobit_corr':C[i]['rmse_censoring_aware'],'exact':O[i]['rmse_exact_only'],'naive':O[i]['rmse_naive'],
  'orig_sigma_bound_seeds':sum(ps['info_censoring_aware']['sigma_at_bound'] for ps in O[i]['per_seed']),
  'corr_sigma_bound_seeds':sum(ps['info_censoring_aware']['sigma_at_bound'] for ps in C[i]['per_seed']),
  'corr_conv':sum(ps['info_censoring_aware']['converged'] for ps in C[i]['per_seed'])} for i in idx]
for r in out['tobit_biggest_moves']: print(r)
# 3 which targets have sigma at bound
sb={}
for l,T in list(L.items())+[('orig',O)]:
    sb[l]={arm:sorted({(t['tid'],t['per_seed'][0]['n_train_exact_only']) for t in T for ps in t['per_seed'] if ps[IK[arm]]['sigma_at_bound']}) for arm in K}
out['sigma_bound_targets']={l:{a:[list(x) for x in v] for a,v in d_.items()} for l,d_ in sb.items()}
print('sigma-bound targets', sb)
# fits with censored z<-7
out['tobit_fits_cens_z_lt_m7']={l:sorted({t['tid'] for t in T for ps in t['per_seed'] if ps['info_censoring_aware'].get('n_cens_z_lt_m7',0)>0}) for l,T in list(L.items())+[('orig',O)]}
print(out['tobit_fits_cens_z_lt_m7'])
# 4 per-arm best-lambda (selected on mean TEST RMSE; optimistic, symmetric)
best={}
for k in K:
    m={l:np.mean([t[k] for t in T]) for l,T in L.items()}
    best[k]=min(m,key=m.get)
out['per_arm_best_lam_global']=best; print('best lam per arm',best)
def contr(a,b):
    da=np.array([t[a] for t in L[best[a]]]); db=np.array([t[b] for t in L[best[b]]]); dd=da-db
    lo,hi=boot_targets(dd); return {'mean':round(float(dd.mean()),4),'ci':[round(lo,4),round(hi,4)],'n_worse':int((dd>0).sum())}
cfrac=np.array([t['censored_frac'] for t in raw])
g={n:contr(a,b) for n,a,b in (('naive_minus_exact','rmse_naive','rmse_exact_only'),('matched_minus_exact','rmse_naive_matched','rmse_exact_only'),('tobit_minus_exact','rmse_censoring_aware','rmse_exact_only'),('tobit_minus_naive','rmse_censoring_aware','rmse_naive'))}
out['contrasts_at_per_arm_global_best_lam']=g; print(g)
# per-target oracle lambda per arm
def orc(k): return np.array([min(L[l][i][k] for l in L) for i in range(len(raw))])
o={n:(orc(a)-orc(b)) for n,a,b in (('naive_minus_exact','rmse_naive','rmse_exact_only'),('matched_minus_exact','rmse_naive_matched','rmse_exact_only'),('tobit_minus_exact','rmse_censoring_aware','rmse_exact_only'),('tobit_minus_naive','rmse_censoring_aware','rmse_naive'))}
out['contrasts_per_target_oracle_lam']={n:{'mean':round(float(v.mean()),4),'ci':[round(x,4) for x in boot_targets(v)],'n_worse':int((v>0).sum())} for n,v in o.items()}
out['contrasts_per_target_oracle_lam']['p3_rho']=round(float(stats.spearmanr(cfrac,o['naive_minus_exact'])[0]),3)
out['contrasts_per_target_oracle_lam']['p3_rho_matched']=round(float(stats.spearmanr(cfrac,o['matched_minus_exact'])[0]),3)
print('oracle',out['contrasts_per_target_oracle_lam'])
# mean RMSE per arm per lam
out['mean_rmse']={l:{k:round(float(np.mean([t[k] for t in T])),4) for k in K} for l,T in L.items()}
json.dump(out,open('s4_sensitivity_extras.json','w'),indent=1)
