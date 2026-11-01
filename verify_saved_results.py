"""Offline saved-artifact verification for PoseSplit; no media download or training.

Replays saved features, small heads, target projection and paired-room statistics.
This is narrower than the independent raw-PCM/WAV audit bundled as an attestation.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
HEADINGS=(0,90,180,270)
SEEDS=(101,211,307,401,503)
ARMS=('pose_aware','canonical_matched_orientation','wrong_pose_negative_control')
CHECKS=collections.Counter()
MAXERR=collections.defaultdict(float)
def load(p):return json.loads(Path(p).read_text())
def check(ok, label):
    if not bool(ok):raise AssertionError(label)
    CHECKS[label]+=1

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def bytehash(b):return hashlib.sha256(b).hexdigest()

def close(a,b,label,tol=2e-9):
    a,b=np.asarray(a,float),np.asarray(b,float)
    check(a.shape==b.shape,label+' shape')
    finite=np.isfinite(a)&np.isfinite(b)
    err=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    MAXERR[label]=max(MAXERR[label],err)
    check(np.allclose(a,b,atol=tol,rtol=tol,equal_nan=True),label)

def normalized(x):
    x=np.asarray(x,float);length=np.linalg.norm(x)
    check(length>1e-12,'nonzero direction')
    return x/length

def world_direction(az,el):
    az,el=math.radians(az),math.radians(el)
    return np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])

def passive_matrix(yaw):
    t=math.radians(yaw);c,s=math.cos(t),math.sin(t)
    return np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])

def frustum(direction,yaw):
    x,y,z=passive_matrix(yaw)@np.asarray(direction)
    return bool(x>0. and abs(y)<=x+1e-9 and abs(z)<=x+1e-9)

def angular(a,b):
    a,b=normalized(a),normalized(b)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a,b))),float(np.dot(a,b))))

def input_vector(feature,relative_yaw):
    t=math.radians(relative_yaw)
    return np.r_[feature,math.cos(t),math.sin(t)]

def forward(weights,x):
    hidden=np.tanh(np.asarray(x)@weights[0]+weights[1])
    logit=np.clip(hidden@weights[2]+weights[3],-40,40)
    return np.asarray(1./(1.+np.exp(-logit))).reshape(-1)

def confusions(gold,prob):
    y=np.asarray(gold,bool);positive=np.asarray(prob)>=.5
    return np.array([np.sum(positive&y),np.sum(~positive&~y),np.sum(positive&~y),np.sum(~positive&y)],dtype=np.int64)

def balanced(c):
    c=np.asarray(c,float);tp,tn,fp,fn=np.moveaxis(c,-1,0)
    return np.where((tp+fn>0)&(tn+fp>0),.5*(tp/np.maximum(tp+fn,1)+tn/np.maximum(tn+fp,1)),np.nan)

def metrics(gold,prob):
    y=np.asarray(gold,bool);p=np.asarray(prob,float);c=confusions(y,p);tp,tn,fp,fn=c
    tpr=float(tp/(tp+fn)) if tp+fn else None
    tnr=float(tn/(tn+fp)) if tn+fp else None
    return {'cases':len(y),'positive':int(y.sum()),'feature_failures':0,'accuracy':float((tp+tn)/len(y)),
        'sensitivity':tpr,'specificity':tnr,'balanced_accuracy':float(balanced(c)) if tpr is not None and tnr is not None else None,
        'brier':float(np.mean((p-y)**2)),
        'log_loss':float(np.mean(-(y*np.log(np.clip(p,1e-12,1))+(~y)*np.log(np.clip(1-p,1e-12,1)))))}

def compare_metrics(actual,expected,label):
    check(set(actual)==set(expected),label+' metric keys')
    for key,value in actual.items():
        target=expected[key]
        if value is None or target is None:check(value is target,label+' undefined '+key)
        else:close(value,target,label+' '+key)

def confidence(v):
    v=np.asarray(v,float);valid=v[np.isfinite(v)]
    return {'confidence95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
       'undefined_replicates':int(np.sum(~np.isfinite(v))),'total_replicates':len(v),'policy':'Absentclass=>null;noredraw.'}

def compare_interval(actual,expected,label):
    for field in ['undefined_replicates','total_replicates','policy']:check(actual[field]==expected[field],label+' '+field)
    if actual['confidence95'] is None or expected['confidence95'] is None:check(actual['confidence95'] is expected['confidence95'],label+' undefined CI')
    else:close(actual['confidence95'],expected['confidence95'],label+' quantiles')

def key(row):return (row['recording'],row['start_frame'],row['regime'],row['heading'])

def verify(run):
    result=load(run/'result.json');frozen=load(run/'FROZEN.json');selection=load(run/'selection.json')
    check(result['status']=='completed','completed result')
    for name,field in [('protocol.json','protocol_sha256'),('selection.json','selection_sha256'),('source_catalog.json','source_catalog_sha256')]:check(digest(run/name)==frozen[field],'frozen '+name)
    for name,expected in result['source_hashes'].items():check(digest(run/name)==expected,'result binding '+name)
    check(len(load(run/'source_readback.json')['files'])==46,'46source-readback metadata records')
    directions={};metadata={}
    for rec in selection:
        for win in rec['windows']:
            raw=win['labels'];check(len(raw)==5,'5MoCap frames per window')
            d=normalized(np.sum([world_direction(r[3],r[4]) for r in raw],axis=0));close(d,win['gold_unit_direction'],'saved MoCap direction')
            directions[(rec['recording'],win['start_frame'])]=d
            metadata[(rec['recording'],win['start_frame'])]=rec
    check(len(directions)==276,'276saved base windows')
    real=load(run/'real_features.json');check(len(real)==2208,'2208saved feature variants')
    indexed={key(r):r for r in real};check(len(indexed)==2208,'unique real variant keys')
    xs=[];keys=list(indexed);gold=[]
    for k,r in indexed.items():
        rec=metadata[(r['recording'],r['start_frame'])];mic=r['heading'] if r['regime']=='co_rig_virtual' else 0
        check(r['mic_yaw']==mic and r['camera_relative_yaw']==r['heading']-mic,'known pose configuration')
        f=np.asarray(r['features'],float);check(f.shape==(4,) and np.isfinite(f).all() and r['feature_error'] is None,'saved acoustic feature contract')
        check(r['gold']==frustum(directions[(r['recording'],r['start_frame'])],r['heading']),'saved target projection')
        check(r['analytic_prediction']==frustum(f[:3],r['camera_relative_yaw']),'analytic fixed-feature baseline')
        for field in ['room','site','split']:check(r[field]==rec[field],'saved variant '+field)
        xs.append(input_vector(f,r['camera_relative_yaw']));gold.append(bool(r['gold']))
    xs=np.asarray(xs);gold=np.asarray(gold,bool);positions={k:i for i,k in enumerate(keys)}
    validation=load(run/'synthetic_validation.json');check(len(validation)==15,'15head metadata records');weights={}
    for item in validation:
        path=run/'models'/f"{item['seed']}_{item['arm']}.npz";check(digest(path)==item['model_sha256'],'saved head SHA')
        with np.load(path,allow_pickle=False) as z:params=[z[f'p{i}'].copy() for i in range(4)]
        check([p.shape for p in params]==[(6,32),(32,),(32,),(1,)] and all(np.isfinite(p).all() for p in params),'head architecture')
        weights[(item['seed'],item['arm'])]=params
    check(len(weights)==15,'15distinct heads')
    synthetic_validation={};source_count=0
    for seed in SEEDS:
        records=load(run/'synthetic'/str(seed)/'manifest.json');check(len(records)==320,'320synthetic feature manifests per seed')
        for split,n in [('train',256),('validation',64)]:
            rows=sorted([r for r in records if r['split']==split],key=lambda r:r['index']);check(len(rows)==n,'synthetic split cardinality')
            correct=[];wrong=[];canonical=[];cy=[];wy=[]
            for r in rows:
                d=normalized(np.array(r['source_position'])-np.array(r['mic_position']));labels=[frustum(d,h) for h in HEADINGS];check(labels==r['labels'] and sum(labels)==1,'synthetic geometric labels')
                f=np.asarray(r['decoded_features'],float);shift=r['wrong_label_shift'];check(shift in [1,2,3],'wrong-label cyclic shift')
                for j,h in enumerate(HEADINGS):
                    correct.append(input_vector(f,h));wrong.append(input_vector(f,h));canonical.append(input_vector(f,0));cy.append(labels[j]);wy.append(labels[(j+shift)%4])
            correct,wrong,canonical=np.asarray(correct),np.asarray(wrong),np.asarray(canonical);cy,wy=np.asarray(cy,bool),np.asarray(wy,bool)
            check(np.array_equal(correct,wrong),'exact saved correct/wrong6Dmultiset')
            check(np.array_equal(correct[:,:4],canonical[:,:4]),'canonical audio multiset')
            check(np.array_equal(cy.reshape(-1,4).sum(1),wy.reshape(-1,4).sum(1)) and np.all(np.any(cy.reshape(-1,4)!=wy.reshape(-1,4),axis=1)),'wrong class multiset and shifted positives')
            if split=='validation':synthetic_validation[seed]=(correct,cy)
        source_count+=len(records)
    check(source_count==1600,'1600synthetic source manifest records')
    for item in validation:
        vx,vy=synthetic_validation[item['seed']];prob=forward(weights[(item['seed'],item['arm'])],vx)
        compare_metrics(metrics(vy,prob),item['metrics'],'saved synthetic validation')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines() if line.strip()];check(len(predictions)==35328,'35328saved predictions')
    model_probs={k:forward(v,xs) for k,v in weights.items()};analytic=np.array([float(indexed[k]['analytic_prediction']) for k in keys]);seen=set();grouped=collections.defaultdict(list)
    for row in predictions:
        k=key(row);identity=(k,row['arm'],row['seed']);check(identity not in seen,'unique prediction keys');seen.add(identity);i=positions[k]
        expected=analytic[i] if row['arm']=='analytic' else model_probs[(row['seed'],row['arm'])][i]
        close(row['probability'],expected,'independent saved head forward probability');check((row['probability']>=.5)==(expected>=.5),'saved threshold decision')
        check(row['gold']==gold[i] and all(row[f]==indexed[k][f] for f in ['split','site','room','recording']),'prediction target/unit binding')
        grouped[(row['arm'],row['seed'],row['regime'])].append((k,bool(gold[i]),float(expected)))
    for k in keys:check((k,'analytic',None) in seen and all((k,arm,seed) in seen for seed in SEEDS for arm in ARMS),'complete16head/source predictions')
    rooms=sorted({r['room'] for r in real if r['split']=='locked'});check(len(rooms)==7,'7locked room clusters')
    saved_boot=load(run/'room_bootstrap_replicates.json');draw=np.asarray(saved_boot['draw_room_indices']);check(saved_boot['room_order']==rooms and np.array_equal(draw,np.random.default_rng(9137).integers(0,7,(2000,7))),'2000paired room draws')
    multiplicity=np.array([np.bincount(r,minlength=7) for r in draw]);boot={};primary={};view_lookup={(r['arm'],r['regime']):r for r in result['view_metrics']}
    for arm in ('analytic',)+ARMS:
        seeds=(None,) if arm=='analytic' else SEEDS
        for regime in ['camera_only','co_rig_virtual']:
            counts=[];pooled_y=[];pooled_p=[]
            for seed in seeds:
                rows=[r for r in grouped[(arm,seed,regime)] if indexed[r[0]]['split']=='locked'];check(len(rows)==672,'672locked views per head/regime')
                per_room=[]
                for room in rooms:
                    values=[r for r in rows if indexed[r[0]]['room']==room];per_room.append(confusions([v[1] for v in values],[v[2] for v in values]))
                counts.append(per_room);pooled_y.extend(v[1] for v in rows);pooled_p.extend(v[2] for v in rows)
            counts=np.asarray(counts);seed_ba=balanced(counts.sum(1));value=float(seed_ba.mean());replicates=np.mean(balanced(np.einsum('br,src->sbc',multiplicity,counts)),axis=0)
            target=view_lookup[(arm,regime)];compare_metrics(metrics(pooled_y,pooled_p),target['pooled_metrics'],'locked pooled statistics');close(value,target['primary_mean_seed_pooled_balanced_accuracy'],'primary BA')
            close(seed_ba,[target['seed_pooled_balanced_accuracy'][str(v)] for v in seeds],'per-seed pooled BA')
            close(replicates,[np.nan if x is None else x for x in saved_boot['estimates'][arm+'/'+regime]],'all2000bootstrap values');compare_interval(confidence(replicates),target['room_cluster_bootstrap'],'bootstrap interval')
            primary[(arm,regime)]=value;boot[(arm,regime)]=replicates
    for target in result['paired_camera_only_room_contrasts']:
        other=target['contrast'].removeprefix('pose_aware_minus_');close(primary[('pose_aware','camera_only')]-primary[(other,'camera_only')],target['paired_difference_of_mean_seed_pooled_BA'],'paired primary difference');compare_interval(confidence(boot[('pose_aware','camera_only')]-boot[(other,'camera_only')]),target['paired_room_cluster_bootstrap'],'paired bootstrap interval')
    for target in load(run/'descriptive_recording_room_metrics.json'):
        rows=[r for r in grouped[(target['arm'],target['seed'],target['regime'])] if indexed[r[0]]['split']=='locked' and indexed[r[0]][target['unit_type']]==target['unit']]
        compare_metrics(metrics([v[1] for v in rows],[v[2] for v in rows]),target['metrics'],'descriptive group statistics');check(confusions([v[1] for v in rows],[v[2] for v in rows]).tolist()==target['confusion_TP_TN_FP_FN'],'descriptive group confusion')
    base=[r for r in real if r['split']=='locked' and r['regime']=='camera_only' and r['heading']==0];check(len(base)==168,'168locked base windows')
    for values,name in [([angular(np.array(r['features'])[:3],directions[(r['recording'],r['start_frame'])]) for r in base],'analytic_audio_real_angular_error'),([angular([1,0,0],directions[(r['recording'],r['start_frame'])]) for r in base],'constant_front_real_angular_error')]:
        actual={'median_deg':float(np.median(values)),'mean_deg':float(np.mean(values)),'within20deg_rate':float(np.mean(np.array(values)<=20))}
        for k,v in actual.items():close(v,result[name][k],name+' '+k,tol=2e-5)
    return {'status':'passed','scope':'Offline saved-feature/head/target/paired-room replay; no training or raw-media download.','base_windows':276,'locked_windows':168,'real_variants':2208,'saved_head_predictions_recomputed':35328,'saved_heads':15,'synthetic_manifest_records':1600,'paired_room_bootstrap_replicates':2000,'room_clusters':7,'primary_balanced_accuracy':{a+'/'+r:v for (a,r),v in primary.items()},'maximum_arithmetic_discrepancies':dict(MAXERR),'checks_completed':sum(CHECKS.values()),'limitations':['This verifier uses saved acoustic features and synthetic decoded-feature manifests; it does not recompute STFT features from absent WAVs or re-decode source media.','The full independent raw-PCM audit is a separate attestation; full source archives, videos and1600synthetic WAVs are intentionally not bundled.','This is audio plus known pose, not learned visual fusion; co-rig reversal and seven-room inference limits remain.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'runs/renewed-09-posesplit-20261031');args=parser.parse_args()
    print(json.dumps(verify(args.results.resolve()),indent=2,allow_nan=False))
if __name__=='__main__':main()
