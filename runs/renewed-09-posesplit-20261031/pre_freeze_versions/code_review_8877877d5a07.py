"""PoseSplit: a new frozen synthetic-to-real audio-plus-known-pose pilot.

Plan/freeze never decode audio/video or train. Run requires reviewed file hashes.
All source/room units, analytic baselines and invalid-data controls are explicit.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,io,json,math,os,subprocess,time,wave,zipfile,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HEADINGS=(0,90,180,270);ARMS=('pose_aware','canonical_matched_orientation','wrong_pose_negative_control')
SEEDS=(101,211,307,401,503);FS=24000;SAMPLES=12000

def sha(x):return hashlib.sha256(x.read_bytes() if isinstance(x,Path) else x).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def unit(v):return v/max(float(np.linalg.norm(v)),1e-12)
def direction(az,el):
 a,e=np.deg2rad([az,el]);return np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
def rot(v,yaw):
 c,s=np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw));return np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1],v[2]])
def visible(v,yaw):
 x,y,z=rot(v,yaw);return bool(x>0 and abs(y)<=x and abs(z)<=x)
def angle(a,b):return float(np.rad2deg(np.arccos(np.clip(np.dot(unit(a),unit(b)),-1,1))))
def audio_rotate(a,yaw):
 b=a.copy();c,s=np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw));b[:,3]=c*a[:,3]+s*a[:,1];b[:,1]=-s*a[:,3]+c*a[:,1];return b

def zip_catalog(url,expected):
 import fsspec,urllib.request
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as response:
  if response.status!=206 or not response.headers.get('Content-Range','').endswith('/'+str(expected)):raise RuntimeError('Require exact HTTP206 and immutable archive length; no full fallback.')
 with fsspec.open(url,block_size=65536,cache_type='readahead') as f:
  assert f.size==expected
  with zipfile.ZipFile(f) as z:return {i.filename:{'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC} for i in z.infolist() if not i.is_dir()}

def eligible(raw):
 frames=collections.defaultdict(list)
 for line in raw.decode().splitlines():
  if line.strip():
   r=[int(x) for x in line.split(',')];assert len(r)==6;frames[r[0]].append(r)
 candidates=[]
 for start in range(0,max(frames,default=-1)+1,5):
  rows=[frames.get(t,[]) for t in range(start,start+5)]
  if not all(len(x)==1 for x in rows):continue
  a=[x[0] for x in rows]
  if len({(r[1],r[2]) for r in a})!=1 or any(abs(r[4])>45 for r in a):continue
  vectors=[direction(r[3],r[4]) for r in a]
  if max(angle(v,w) for v in vectors for w in vectors)>15+1e-9:continue
  candidates.append({'start_frame':start,'seconds':start/10,'class':a[0][1],'source':a[0][2],'labels':a,'gold_unit_direction':unit(np.mean(vectors,axis=0)).tolist()})
 return candidates

def plan(out):
 if out.exists() and any(out.iterdir()):raise FileExistsError('Plan into a fresh directory.')
 p=read(ROOT/'work/task09_protocol.json');metadata=ROOT/p['sources']['metadata'];assert hashlib.md5(metadata.read_bytes()).hexdigest()==p['sources']['metadata_md5']
 catalogs={k:zip_catalog(p['sources'][k+'_url'],p['sources'][k+'_archive_bytes']) for k in ['foa','video']}
 grouped=collections.defaultdict(list);ledger=[]
 with zipfile.ZipFile(metadata) as z:
  for name in sorted(x for x in z.namelist() if x.endswith('.csv')):
   ident=Path(name).stem;parts=ident.split('_');room=parts[1];split='development' if 'dev-train-' in name else 'locked';site='sony' if '-sony/' in name else 'tau'
   am=name.replace('metadata_dev/','foa_dev/').replace('.csv','.wav');vm=name.replace('metadata_dev/','video_dev/').replace('.csv','.mp4')
   windows=eligible(z.read(name));reason='eligible' if len(windows)>=8 and am in catalogs['foa'] and vm in catalogs['video'] else 'too_few_eligible_windows_or_missing_audio_video'
   ledger.append({'recording':ident,'room':room,'split':split,'eligible_windows':len(windows),'status':reason})
   if reason=='eligible':grouped[(split,room)].append({'recording':ident,'room':room,'split':split,'site':site,'metadata_member':name,'audio_member':am,'video_member':vm,'audio_catalog':catalogs['foa'][am],'video_catalog':catalogs['video'][vm],'windows':windows})
 assert len({r for s,r in grouped if s=='development'})==9 and len({r for s,r in grouped if s=='locked'})==7,'Expected all9development/7lockedrooms.'
 selected=[]
 for (split,room),rows in sorted(grouped.items()):
  quota=1 if split=='development' else 2
  if len(rows)<quota:raise RuntimeError(f'Room quota unmet before decoding: {split} {room}')
  for row in sorted(rows,key=lambda r:r['recording'])[:quota]:
   row['eligible_windows_count']=len(row['windows']);row['windows']=sorted(sorted(row['windows'],key=lambda w:sha(f"PoseSplit-v1:20261031:{row['recording']}:{w['start_frame']}".encode()))[:12],key=lambda w:w['start_frame']);selected.append(row)
 assert len(selected)==23 and sum(r['split']=='locked' for r in selected)==14
 total=sum(r[k]['compressed_bytes'] for r in selected for k in ['audio_catalog','video_catalog']);assert total<=p['sources']['total_source_member_download_cap_bytes']
 out.mkdir(parents=True);write(out/'protocol.json',p);write(out/'selection.json',selected);write(out/'selection_ledger.json',ledger);write(out/'source_catalog.json',catalogs)
 write(out/'plan_readback.json',{'status':'labels_only_plan_complete','source_member_compressed_bytes':total,'recordings':len(selected),'development_recordings':9,'locked_recordings':14,'development_rooms':9,'locked_rooms':7,'windows':dict(collections.Counter(r['split'] for r in selected for _ in r['windows'])),'code_sha256':sha(Path(__file__)),'selection_sha256':sha(out/'selection.json'),'protocol_sha256':sha(out/'protocol.json'),'no_audio_video_features_decoded':True,'no_training_or_scoring':True})
 print(json.dumps(read(out/'plan_readback.json'),indent=2))

def freeze(out):
 assert not (out/'FROZEN.json').exists();p=read(out/'plan_readback.json');assert p['code_sha256']==sha(Path(__file__))
 write(out/'FROZEN.json',{'status':'frozen_after_root_and_independent_review','reviewer':'Root must invoke this stage only after both reviews approve.','code_sha256':sha(Path(__file__)),'protocol_sha256':sha(out/'protocol.json'),'selection_sha256':sha(out/'selection.json'),'source_catalog_sha256':sha(out/'source_catalog.json'),'frozen_unix':time.time()});print('Frozen protocol/source/selection hashes.')

def verify_freeze(out):
 f=read(out/'FROZEN.json');assert f['status']=='frozen_after_root_and_independent_review' and f['code_sha256']==sha(Path(__file__))
 for name,key in [('protocol.json','protocol_sha256'),('selection.json','selection_sha256'),('source_catalog.json','source_catalog_sha256')]:assert sha(out/name)==f[key]

def fetch(out):
 verify_freeze(out);import fsspec
 p=read(out/'protocol.json');selection=read(out/'selection.json');records=[]
 for kind in ['foa','video']:
  url=p['sources'][kind+'_url'];key='audio' if kind=='foa' else 'video'
  with fsspec.open(url,block_size=65536,cache_type='readahead') as remote:
   assert remote.size==p['sources'][kind+'_archive_bytes']
   with zipfile.ZipFile(remote) as z:
    for row in selection:
     member=row[key+'_member'];expected=row[key+'_catalog'];dest=out/'sources'/member
     if dest.exists():raw=dest.read_bytes()
     else:
      raw=z.read(member);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
     assert len(raw)==expected['bytes'] and zlib.crc32(raw)==expected['crc32'];records.append({'member':member,'sha256':sha(raw),'bytes':len(raw),'crc32':expected['crc32']})
 write(out/'source_readback.json',{'status':'passed','files':records});print('Fetched and verified46 selected audio/video source members.')

def read_window(path,seconds):
 with wave.open(str(path),'rb') as w:
  assert w.getnchannels()==4 and w.getframerate()==FS and w.getsampwidth()==2 and w.getcomptype()=='NONE'
  w.setpos(round(seconds*FS));raw=w.readframes(SAMPLES);assert len(raw)==SAMPLES*4*2
 return np.frombuffer(raw,dtype='<i2').reshape(-1,4).astype(np.float64)/32768.,sha(raw)

def emit(path,a):
 a=np.asarray(np.rint(np.clip(a,-.999,.999)*32767),dtype='<i2')
 with wave.open(str(path),'wb') as w:w.setnchannels(4);w.setsampwidth(2);w.setframerate(FS);w.writeframes(a.tobytes())
 return read_window(path,0)[0]

def features(a):
 assert a.shape==(SAMPLES,4) and np.all(np.isfinite(a));starts=range(0,SAMPLES-512+1,256)
 spec=np.fft.rfft(np.stack([a[i:i+512]*np.hanning(512)[:,None] for i in starts]),axis=1);freq=np.fft.rfftfreq(512,1/FS);spec=spec[:,(freq>=300)&(freq<=4500),:]
 w=spec[:,:,0];xyz=spec[:,:,[3,1,2]];iv=np.real(w.conj()[:,:,None]*xyz).sum(axis=(0,1));norm=float(np.linalg.norm(iv));denom=np.sqrt(float((abs(w)**2).sum()*(abs(xyz)**2).sum()));coherence=float(np.clip(norm/max(denom,1e-12),0,1))
 if norm<=1e-12:raise ValueError('No measurable active intensity in selected window.')
 return np.r_[iv/norm,coherence]

def head_input(f,delta):return np.r_[f,np.cos(np.deg2rad(delta)),np.sin(np.deg2rad(delta))]

def synthetic_audio(rng):
 room=np.r_[rng.uniform(4,8,2),rng.uniform(2.5,3.5)];mic=np.r_[room[:2]/2,1.3]
 while True:
  az=float(rng.uniform(-180,180));el=float(rng.uniform(-25,25))
  if min(abs(((az-b+180)%360)-180) for b in [-135,-45,45,135])>=1:break
 d=direction(az,el);distance=float(rng.uniform(.4,1.2));source=mic+distance*d;assert np.all(source>0) and np.all(source<room)
 noise=rng.normal(size=SAMPLES);spectrum=np.fft.rfft(noise);freq=np.fft.rfftfreq(SAMPLES,1/FS);tilt=float(rng.uniform(-1,1));spectrum*=np.where((freq>=100)&(freq<=6000),np.maximum(freq,100)**tilt,0);dry=np.fft.irfft(spectrum,n=SAMPLES);dry/=max(np.std(dry),1e-9)
 t=np.arange(SAMPLES)/FS;dry+=rng.uniform(0,.5)*np.sin(2*np.pi*rng.uniform(300,2500)*t+rng.uniform(-np.pi,np.pi));dry*=.7+.3*np.sin(2*np.pi*rng.uniform(1,5)*t+rng.uniform(-np.pi,np.pi))
 a=dry[:,None]*np.array([1,d[1],d[2],d[0]])[None,:];paths=[]
 for axis in range(3):
  for side in [0,1]:
   mirror=source.copy();mirror[axis]=-source[axis] if side==0 else 2*room[axis]-source[axis];vector=mirror-mic;length=float(np.linalg.norm(vector));direct=vector/length;delay=round((length-distance)/343*FS);reflection=float(rng.uniform(.15,.65));gain=reflection*distance/length
   if delay<SAMPLES:a[delay:]+=gain*dry[:SAMPLES-delay,None]*np.array([1,direct[1],direct[2],direct[0]])
   paths.append({'axis':axis,'side':side,'image_source':mirror.tolist(),'path_distance':length,'relative_delay_samples':delay,'reflection':reflection})
 snr=float(rng.uniform(10,30));a+=rng.normal(size=a.shape)*np.sqrt(np.mean(a*a))*10**(-snr/20);gain=.45/max(float(np.max(abs(a))),1e-12);a*=gain
 return a,d,{'room_dimensions':room.tolist(),'mic_position':mic.tolist(),'source_position':source.tolist(),'azimuth':az,'elevation':el,'snr_db':snr,'global_gain':gain,'reflection_paths':paths}

def synthetic_dataset(out,seed):
 rng=np.random.default_rng(seed);groups={};metadata=[]
 for split,n in [('train',256),('validation',64)]:
  rows=[]
  for i in range(n):
   audio,d,meta=synthetic_audio(rng);path=out/'synthetic'/str(seed)/split/f'{i:04}.wav';path.parent.mkdir(parents=True,exist_ok=True);decoded=emit(path,audio);f=features(decoded);labels=[visible(d,h) for h in HEADINGS];assert sum(labels)==1
   shift=int(rng.integers(1,4));metadata.append(dict(meta,seed=seed,split=split,index=i,waveform=str(path.relative_to(out)),sha256=sha(path),labels=labels,wrong_label_shift=shift,decoded_features=f.tolist()));rows.append((f,labels,shift))
  groups[split]=rows
 write(out/'synthetic'/str(seed)/'manifest.json',metadata);return groups

def arm_cases(groups,arm):
 xs=[];ys=[]
 for f,labels,shift in groups:
  for j,h in enumerate(HEADINGS):
   xs.append(head_input(f,0 if arm=='canonical_matched_orientation' else h));ys.append(labels[0] if arm=='canonical_matched_orientation' else labels[(j+shift)%4] if arm=='wrong_pose_negative_control' else labels[j])
 return np.asarray(xs),np.asarray(ys,dtype=float)

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def predict(params,x):return sigmoid(np.tanh(x@params[0]+params[1])@params[2]+params[3])
def fit(x,y,seed):
 rng=np.random.default_rng(seed);params=[rng.normal(0,1/np.sqrt(6),(6,32)),np.zeros(32),rng.normal(0,1/np.sqrt(32),32),np.zeros(1)];m=[np.zeros_like(v) for v in params];v=[np.zeros_like(q) for q in params];history=[]
 for step in range(1,1001):
  h=np.tanh(x@params[0]+params[1]);prob=sigmoid(h@params[2]+params[3]);dz=(prob-y)/len(y);dh=dz[:,None]*params[2][None,:]*(1-h*h)
  grads=[x.T@dh+1e-4*params[0],dh.sum(axis=0),h.T@dz+1e-4*params[2],np.array([dz.sum()])]
  for k,g in enumerate(grads):
   m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g;params[k]-=.01*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
  if step in [1,100,500,1000]:
   loss=float(np.mean(-y*np.log(np.clip(prob,1e-12,1))-(1-y)*np.log(np.clip(1-prob,1e-12,1))))
   history.append({'step':step,'binary_crossentropy':loss})
 return params,history

def metrics(rows):
 y=np.array([r['gold'] for r in rows],dtype=bool);p=np.array([.5 if r['probability'] is None else r['probability'] for r in rows]);valid=np.array([r['probability'] is not None for r in rows]);pred=p>=.5;correct=(pred==y)&valid
 tpr=float((correct&y).sum()/y.sum()) if y.any() else None;tnr=float((correct&~y).sum()/(~y).sum()) if (~y).any() else None
 return {'cases':len(rows),'positive':int(y.sum()),'feature_failures':int((~valid).sum()),'accuracy':float(correct.mean()),'sensitivity':tpr,'specificity':tnr,'balanced_accuracy':(tpr+tnr)/2 if tpr is not None and tnr is not None else None,'brier':float(np.mean(np.where(valid,(p-y)**2,1.))),'log_loss':float(np.mean(np.where(valid,-(y*np.log(np.clip(p,1e-12,1))+(~y)*np.log(np.clip(1-p,1e-12,1))),-np.log(1e-12))))}

def natural_features(out):
 rows=[];selection=read(out/'selection.json')
 for rec in selection:
  for win in rec['windows']:
   item={k:rec[k] for k in ['recording','room','split','site']}|win;path=out/'sources'/rec['audio_member'];window_rows=[]
   try:
    a,rawsha=read_window(path,win['seconds']);basebytes=np.asarray(a,dtype='<f4').tobytes();basehash=sha(basebytes);item.update(source_pcm_sha256=rawsha,float_pcm_sha256=basehash)
    for regime in ['camera_only','co_rig_virtual']:
     for heading in HEADINGS:
      local=audio_rotate(a,heading) if regime=='co_rig_virtual' else a;pcm=np.asarray(local,dtype='<f4').tobytes();decoded=np.frombuffer(pcm,dtype='<f4').reshape(-1,4).astype(np.float64);f=features(decoded);mic=heading if regime=='co_rig_virtual' else 0;relative=heading-mic;true_mic=rot(np.array(win['gold_unit_direction']),mic);analytic=visible(f[:3],relative)
      window_rows.append(item|{'regime':regime,'heading':heading,'mic_yaw':mic,'camera_relative_yaw':relative,'pcm_sha256':sha(pcm),'features':f.tolist(),'gold':visible(np.array(win['gold_unit_direction']),heading),'analytic_prediction':analytic,'feature_error':None})
    assert len(window_rows)==8
   except Exception as exc:
    failure=item|{'error':type(exc).__name__+':'+str(exc),'phase':'real_feature_validation_before_training','unix':time.time()}
    with (out/'feature_failures.jsonl').open('a') as stream:stream.write(json.dumps(failure)+'\n')
    write(out/'FAILED_BEFORE_TRAINING.json',failure)
    raise RuntimeError('Selected source feature failure; study halted before training/scoring. Preserve logs and IDs.') from exc
   rows.extend(window_rows)
 assert len(rows)==sum(len(r['windows']) for r in selection)*8
 write(out/'real_features.json',rows);write(out/'feature_validation.json',{'status':'passed','windows':len(rows)//8,'variants':len(rows),'training_started':False});return rows

def crop_frame(panorama,yaw,size=320):
 yy,xx=np.mgrid[0:size,0:size];left=-(xx+.5-size/2)/(size/2);up=-(yy+.5-size/2)/(size/2);forward=np.ones_like(left);theta=np.deg2rad(yaw);wx=np.cos(theta)*forward-np.sin(theta)*left;wy=np.sin(theta)*forward+np.cos(theta)*left
 lon=np.arctan2(wy,wx);lat=np.arctan2(up,np.sqrt(wx*wx+wy*wy));h,w=panorama.shape[:2];u=((.5-lon/(2*np.pi))*w-.5)%w;v=np.clip((.5-lat/np.pi)*h-.5,0,h-1);u0=np.floor(u).astype(int);v0=np.floor(v).astype(int);du=(u-u0)[...,None];dv=(v-v0)[...,None]
 out=(1-dv)*((1-du)*panorama[v0,u0]+du*panorama[v0,(u0+1)%w])+dv*((1-du)*panorama[np.minimum(v0+1,h-1),u0]+du*panorama[np.minimum(v0+1,h-1),(u0+1)%w]);return np.clip(np.rint(out),0,255).astype(np.uint8)

def media_audit(out,real):
 from PIL import Image,ImageDraw
 records=[];examples={split:sorted([r for r in read(out/'selection.json') if r['split']==split],key=lambda r:r['recording'])[0]['recording'] for split in ['development','locked']}
 for rec in read(out/'selection.json'):
  win=rec['windows'][0];image_paths=[]
  for offset in [.05,.25,.45]:
   second=win['seconds']+offset;raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(second),'-i',str(out/'sources'/rec['video_member']),'-frames:v','1','-f','image2pipe','-vcodec','png','-']);pan=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
   for heading in HEADINGS:
    rgb=crop_frame(pan,heading);path=out/'media_audit'/rec['recording']/f'{offset:.2f}_{heading:03}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(path)
    matches=[r for r in real if r['recording']==rec['recording'] and r['start_frame']==win['start_frame'] and r['heading']==heading]
    camera=next(r for r in matches if r['regime']=='camera_only');rig=next(r for r in matches if r['regime']=='co_rig_virtual');records.append({'recording':rec['recording'],'split':rec['split'],'seconds':second,'heading':heading,'pixel_rgb_sha256':sha(rgb.tobytes()),'pixel_same_both_regimes':True,'camera_only_pcm_sha256':camera['pcm_sha256'],'co_rig_pcm_sha256':rig['pcm_sha256'],'image':str(path.relative_to(out))})
    if offset==.25:image_paths.append(path)
  if rec['recording']==examples[rec['split']]:
   sheet=Image.new('RGB',(640,700),'white');draw=ImageDraw.Draw(sheet);draw.text((8,5),rec['recording']+' - geometric viewport audit',fill='black')
   for i,path in enumerate(image_paths):sheet.paste(Image.open(path),(320*(i%2),30+335*(i//2)));draw.text((8+320*(i%2),352+335*(i//2)),f'camera yaw {HEADINGS[i]} deg',fill='black')
   sheet.save(out/'media_audit'/f"montage_{rec['split']}.png")
 write(out/'media_audit.json',records)

def confusion(rows):
 y=np.array([r['gold'] for r in rows],dtype=bool);p=np.array([r['probability']>=.5 for r in rows],dtype=bool)
 return np.array([np.sum(p&y),np.sum((~p)&(~y)),np.sum(p&(~y)),np.sum((~p)&y)],dtype=float)

def ba_from_counts(c):
 tp,tn,fp,fn=np.moveaxis(np.asarray(c,dtype=float),-1,0);positive=tp+fn;negative=tn+fp
 return np.where((positive>0)&(negative>0),.5*(tp/np.maximum(positive,1)+tn/np.maximum(negative,1)),np.nan)

def interval(values):
 values=np.asarray(values);good=values[np.isfinite(values)]
 return {'confidence95':np.quantile(good,[.025,.975]).tolist() if len(good) else None,'undefined_replicates':int(np.sum(~np.isfinite(values))),'total_replicates':len(values),'policy':'Absentclass=>null;noredraw.'}

def summarize(out,predictions,real):
 aggregate=[];locked=[r for r in predictions if r['split']=='locked'];rooms=sorted({r['room'] for r in locked});assert len(rooms)==7
 draw=np.random.default_rng(9137).integers(0,7,size=(2000,7));boot_by_arm={};primary_by_arm={};descriptive=[]
 for arm in ['analytic']+list(ARMS):
  seeds=[None] if arm=='analytic' else list(SEEDS)
  for regime in ['camera_only','co_rig_virtual']:
   armrows=[r for r in locked if r['arm']==arm and r['regime']==regime];tensor=np.array([[confusion([r for r in armrows if r['room']==room and r['seed']==seed]) for room in rooms] for seed in seeds]);pooled=tensor.sum(axis=1);seed_ba=ba_from_counts(pooled)
   assert np.all(np.isfinite(seed_ba)),'Primarypooledcohortmustcontainbothclasses.'
   # Same cluster draw for every arm and seed, retaining all cases and multiplicity.
   bootstrap_counts=tensor[:,draw,:].sum(axis=2);bootstrap_ba=ba_from_counts(bootstrap_counts);boot=np.mean(bootstrap_ba,axis=0);primary=float(np.mean(seed_ba));boot_by_arm[(arm,regime)]=boot;primary_by_arm[(arm,regime)]=primary
   for seed in seeds:
    for unit_name in ['room','recording']:
     for ident in sorted({r[unit_name] for r in armrows}):
      selected=[r for r in armrows if r['seed']==seed and r[unit_name]==ident];m=metrics(selected);counts=confusion(selected).astype(int).tolist();descriptive.append({'arm':arm,'seed':seed,'regime':regime,'unit_type':unit_name,'unit':ident,'confusion_TP_TN_FP_FN':counts,'metrics':m})
   aggregate.append({'arm':arm,'regime':regime,'pooled_metrics':metrics(armrows),'seed_pooled_balanced_accuracy':dict(zip([str(v) for v in seeds],seed_ba.tolist())),'primary_mean_seed_pooled_balanced_accuracy':primary,'room_cluster_bootstrap':interval(boot),'statistical_units':'7heldoutroomclusters;allcasespooledwithinseedthenaveragedacross5seeds;views/windows/seedsnotindependent.'})
 differences=[]
 for other in ['canonical_matched_orientation','wrong_pose_negative_control','analytic']:
  key=('pose_aware','camera_only');ref=(other,'camera_only');differences.append({'contrast':'pose_aware_minus_'+other,'paired_difference_of_mean_seed_pooled_BA':primary_by_arm[key]-primary_by_arm[ref],'paired_room_cluster_bootstrap':interval(boot_by_arm[key]-boot_by_arm[ref])})
 write(out/'descriptive_recording_room_metrics.json',descriptive)
 base=[r for r in real if r['split']=='locked' and r['regime']=='camera_only' and r['heading']==0];front=[angle(direction(0,0),np.array(r['gold_unit_direction'])) for r in base]
 result={'status':'completed','study':'PoseSplit','primary_task':'Audio+knownsensorposegeometricviewportavailability;notlearnedAVfusion','model_calls':0,'locked_recordings':14,'locked_rooms':7,'locked_windows':len(base),'analytic_audio_real_angular_error':{'median_deg':float(np.median([r['audio_angular_error_deg'] for r in base])),'mean_deg':float(np.mean([r['audio_angular_error_deg'] for r in base])),'within20deg_rate':float(np.mean([r['within20deg'] for r in base]))},'constant_front_real_angular_error':{'median_deg':float(np.median(front)),'mean_deg':float(np.mean(front)),'within20deg_rate':float(np.mean(np.array(front)<=20))},'view_metrics':aggregate,'paired_camera_only_room_contrasts':differences,'scope':['Procedural4channelwaveformtraining;externalrealhuman/MoCaplabelsheldoutbyroom.','Analyticbaselinecanoutperformlearnedhead;noclaimnewlocalizationalgorithm.','Cameraonly/co-rigtransformvalidityisseparatefromrealDOAtransfer.','Noimagefeatures,physicalocclusion,learnedvisualgenerationorfullAVperceptiongain.','Allviews/seeds/windowsnestedwithinrecordingsandrooms;7roomsallowonlysmall-sampledescriptiveinference.'],'source_hashes':{n:sha(out/n) for n in ['FROZEN.json','protocol.json','selection.json','source_readback.json','real_features.json','predictions.jsonl']}}
 write(out/'result.json',result);return result

def run(out):
 verify_freeze(out)
 if (out/'result.json').exists() or (out/'predictions.jsonl').exists():raise FileExistsError('Preserve observed study; no silent rerun.')
 assert read(out/'source_readback.json')['status']=='passed';started=time.time()
 for item in read(out/'source_readback.json')['files']:assert sha(out/'sources'/item['member'])==item['sha256']
 # Cardinal and pose roundtrips are software checks, not transfer evidence.
 for az in HEADINGS:
  d=direction(az,0)
  for yaw in HEADINGS:assert angle(rot(rot(d,yaw),-yaw),d)<1e-5
 # Emitted/decoded cardinal waveforms validate channel/sign conventions before real data.
 fixture_rows=[]
 for az in HEADINGS:
  d=direction(az,0);t=np.arange(SAMPLES)/FS;waveform=.2*np.sin(2*np.pi*1300*t);a=waveform[:,None]*np.array([1,d[1],d[2],d[0]])
  path=out/'coordinate_fixtures'/f'{az}.wav';path.parent.mkdir(exist_ok=True);decoded=emit(path,a);f=features(decoded);assert angle(f[:3],d)<.01
  for yaw in HEADINGS:assert angle(features(audio_rotate(decoded,yaw))[:3],rot(d,yaw))<.01
  fixture_rows.append({'azimuth':az,'decoded_pcm_sha256':sha(path),'feature_direction':f[:3].tolist(),'angular_error_deg':angle(f[:3],d)})
 write(out/'coordinate_fixture_readback.json',{'status':'passed','scope':'Software/knownwaveformgeometryfixture,notrealtransfer.','rows':fixture_rows})
 real=natural_features(out)
 try:media_audit(out,real)
 except Exception as exc:
  failure={'phase':'real_media_validation_before_training','error':type(exc).__name__+':'+str(exc),'unix':time.time()};write(out/'FAILED_BEFORE_TRAINING.json',failure)
  with (out/'feature_failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
  raise RuntimeError('Media validation failed before training/scoring; preserve IDs/logs.') from exc
 # Every selected audio window and media frame passed before any efficacy scoring/fitting.
 for row in real:
  true_mic=rot(np.array(row['gold_unit_direction']),row['mic_yaw']);row['audio_angular_error_deg']=angle(np.array(row['features'])[:3],true_mic);row['within20deg']=row['audio_angular_error_deg']<=20
 write(out/'real_features.json',real)
 models={};validation=[]
 for seed in SEEDS:
  data=synthetic_dataset(out,seed)
  correct_x,correct_y=arm_cases(data['train'],'pose_aware');wrong_x,_=arm_cases(data['train'],'wrong_pose_negative_control');assert np.array_equal(correct_x,wrong_x)
  for arm in ARMS:
   x,y=arm_cases(data['train'],arm);params,history=fit(x,y,seed);modelpath=out/'models'/f'{seed}_{arm}.npz';modelpath.parent.mkdir(exist_ok=True);np.savez(modelpath,**{f'p{i}':p for i,p in enumerate(params)});models[(seed,arm)]=params
   vx,vy=arm_cases(data['validation'],'pose_aware');probs=predict(params,vx);validation.append({'seed':seed,'arm':arm,'rows':len(vy),'metrics':metrics([{'gold':bool(g),'probability':float(p)} for g,p in zip(vy,probs)]),'training_history':history,'model_sha256':sha(modelpath)})
 write(out/'synthetic_validation.json',validation);predictions=[]
 for r in real:
  common={k:r[k] for k in ['recording','room','split','site','start_frame','regime','heading','gold','feature_error']};predictions.append(common|{'arm':'analytic','seed':None,'probability':float(r['analytic_prediction']) if r['analytic_prediction'] is not None else None})
  for (seed,arm),params in models.items():
   prob=float(np.asarray(predict(params,head_input(np.array(r['features']),r['camera_relative_yaw']))).reshape(-1)[0]) if r['features'] is not None else None;predictions.append(common|{'arm':arm,'seed':seed,'probability':prob})
 with (out/'predictions.jsonl').open('w') as f:
  for r in predictions:f.write(json.dumps(r,allow_nan=False)+'\n')
 result=summarize(out,predictions,real)
 import resource
 write(out/'execution_resources.json',{'wall_seconds':time.time()-started,'maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'platform':os.uname().sysname,'mac_maxrss_units':'bytesonmacOS;kilobytesonLinux','target48GB':'Boundedstreamingdesign;actualhostRAMnotanexact48GBdevicevalidation.','native_model_calls':0});print(json.dumps({k:result[k] for k in ['status','locked_recordings','locked_rooms','locked_windows','analytic_audio_real_angular_error']},indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','freeze','fetch','run'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 {'plan':plan,'freeze':freeze,'fetch':fetch,'run':run}[a.stage](out)
if __name__=='__main__':main()
