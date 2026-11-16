"""Refit research candidates on all labeled train systems; never score held-outs."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import torch
from torch import nn
from src.paper2.run_hito0_baselines import fit_ridge, predict_ridge
from src.modeling.train_ridge_summary_outer import summarize_sequence
from run_hito0_gru_rmsd import ResidualGRU,prepare,predict,fit_model,setup_seed,protein_errors

run=Path(sys.argv[1])
for name,expected in json.loads((run/'manifest.json').read_text())['files'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected,name
with (run/'train_future_rmsd_rows.csv').open(newline='') as f:
    rows=sorted([r for r in csv.DictReader(f) if r['future_valid'].lower()=='true'],key=lambda r:r['sample_id'])
with np.load(run/'dataset.npz',allow_pickle=False) as z:
    ids=[s.decode() if isinstance(s,bytes) else str(s) for s in z['sample_ids']]
    index={s:i for i,s in enumerate(ids)}
    x=z['X_raw'][[index[r['sample_id']] for r in rows]].astype(np.float64)
y=np.array([float(r['future_target_A']) for r in rows])
systems=np.array([r['system_id'] for r in rows])
groups=np.array([r['group_id'] for r in rows])
seed=20271007
order=sorted(set(groups),key=lambda g:(hashlib.sha256(f'{seed}:{g}'.encode()).digest(),g))
held=order[:max(5,round(.2*len(order)))]
ev=np.flatnonzero(np.isin(groups,held)); tr=np.flatnonzero(~np.isin(groups,held))
assert set(groups[ev]).isdisjoint(groups[tr])
mean,std,scale=prepare(x,y,tr)
setup_seed(seed)
model=ResidualGRU(32)
opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
features=torch.as_tensor(((x[tr]-mean)/std).astype(np.float32))
residual=torch.as_tensor(((y[tr]-x[tr,-1,0])/scale).astype(np.float32))
best_epoch,best_score,stale=0,float('inf'),0
for epoch in range(151):
    if epoch:
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=(model(features)-residual).square().mean()
        assert torch.isfinite(loss)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    score=float(np.mean(protein_errors(y[ev],predict(model,x,ev,mean,std,scale),systems[ev])))
    if score<best_score-1e-5:
        best_epoch,best_score,stale=epoch,score,0
    else:
        stale+=1
    if stale>=20:
        break
all_indices=np.arange(len(y))
mean,std,scale=prepare(x,y,all_indices)
model=fit_model(x,y,all_indices,mean,std,scale,seed,best_epoch,32)
out=run/'full_train'
out.mkdir(exist_ok=False)
torch.save({'state_dict':model.state_dict(),'feature_mean':mean,'feature_std':std,
            'residual_scale':scale,'hidden_size':32,'selected_epoch':best_epoch},out/'gru_model.pt')
summary=summarize_sequence(x)
scores={}
bins=np.array_split(np.asarray(order),4)
for alpha in (.01,.1,1.,10.,100.):
    errors=[]
    for held in bins:
        evaluation=np.isin(groups,held)
        fitted=fit_ridge(summary[~evaluation],y[~evaluation],alpha)
        errors.extend(protein_errors(y[evaluation],predict_ridge(fitted,summary[evaluation]),systems[evaluation]))
    scores[str(alpha)]=float(np.mean(errors))
alpha=min((.01,.1,1.,10.,100.),key=lambda a:(scores[str(a)],a))
fm,fs,ym,ys,beta=fit_ridge(summary,y,alpha)
np.savez(out/'ridge_model.npz',mean=fm,std=fs,y_mean=ym,y_std=ys,coefficients=beta,alpha=alpha)
(out/'metadata.json').write_text(json.dumps({'research_candidate_only':True,
    'n_systems':len(set(systems)),'n_groups':len(order),'n_replicas':len(y),
    'seed':seed,'gru_epoch':best_epoch,'ridge_alpha':alpha,
    'gru_inner_groups':sorted(set(groups[ev])),'ridge_inner_scores':scores,
    'validation_test_evaluated':False,'HITO0_passed':False},indent=2)+'\n')
print(f'FULL_TRAIN_REFIT=PASS gru_epoch={best_epoch} ridge_alpha={alpha}')
