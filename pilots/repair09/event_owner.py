"""EventOwner: new joint pixels/waveform feasibility study, not ContactTrace.

Metadata is for rendering/audit only. score_packet() receives pixels and samples,
not source geometry, schedules, owner labels or seed. Confirmation is a separate
stage requiring a frozen protocol and source-code hash.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, resource, subprocess, sys, time, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
FS=16000; FPS=24; T=96; H=72; W=128; B=24
AZ=np.linspace(-65,65,B); METHODS=('temporal','pooled_spatial','broadband_joint','factorized_joint')

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(x): return hashlib.sha256(x.read_bytes() if isinstance(x,Path) else x).hexdigest()
def corr(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1);a=a-a.mean();b=b-b.mean()
 return float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))
def wav_write(p,a):
 p.parent.mkdir(parents=True,exist_ok=True)
 with wave.open(str(p),'wb') as f:
  f.setparams((4,2,FS,0,'NONE','not compressed'));f.writeframes(np.rint(np.clip(a,-.99,.99)*32767).astype('<i2').tobytes())
def wav_read(p):
 with wave.open(str(p),'rb') as f:
  assert f.getparams()[:3]==(4,2,FS);return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,4).astype(float)/32768

def spec_scene(seed,index,family):
 rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
 # Predeclared repeating strata; held-out families all contain every stratum.
 stratum=('separated','overlap','near_colocated','occluded')[index%4]
 if stratum=='near_colocated': az=np.array([-2,2])+rng.uniform(-7,7)
 else: az=np.array([-1,1])*rng.uniform(22,44,2)+rng.uniform(-4,4)
 n=3 if family.endswith('three-blade') else 4 if family.endswith('four-blade') else 5 if family.endswith('five-blade') else 6
 if 'asymmetric' in family:n=5
 t=np.arange(T)/FPS;env=[]
 shared_centers=np.sort(rng.uniform(.45,3.55,3))
 overlap_lead=float(rng.choice([-1,1]))
 for j in range(2):
  # No fixed global event clock or leading-source order across scenes.
  if stratum!='overlap':centers=np.sort(rng.uniform(.35,3.65,3))
  else:centers=np.clip(shared_centers+(j-.5)*overlap_lead*.09+rng.uniform(-.03,.03,3),.25,3.75)
  widths=rng.uniform(.14,.32,3)
  drive=.12+sum(np.exp(-.5*((t-c)/w)**2)*rng.uniform(.55,1.0) for c,w in zip(centers,widths))
  env.append(drive/np.sqrt(np.mean(drive**2)))
 return {'index':index,'family':family,'stratum':stratum,'seed':seed,
  'azimuths':az.tolist(),'blade_count':n,'radii':rng.uniform(10,13,2).tolist(),
  'y_centers':([27,48] if stratum=='near_colocated' else rng.uniform(34,39,2).tolist()),
  'colors':rng.integers(90,235,(2,3)).tolist(),'background':rng.integers(15,55,3).tolist(),
  'phases':rng.uniform(0,2*np.pi,2).tolist(),'envelopes':np.asarray(env).tolist(),
  'frequency_centers':(np.array([650,2200])+rng.uniform(-100,100,2)).tolist(),
  'snr_db':float(rng.uniform(15,30)),'reflection_gain':float(rng.uniform(.08,.25)),
  'reflection_delay':int(rng.integers(100,400)),
  'common_rig_world_yaw':float(rng.uniform(-180,180)),
  'model_scope':'Stylized shared-state rotor kinematics and source audio, not realistic fan-acoustic simulation.'}

def render_video(spec,assignment):
 rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],spec['index'],701]))
 env=np.asarray(spec['envelopes']);speed=2+5*env;angle=np.cumsum(speed,axis=1)/FPS+np.asarray(spec['phases'])[:,None]
 # Different owner assignments use same sampled camera sensor noise.
 frames=[];base=np.empty((H,W,3),dtype=np.uint8);base[:]=spec['background']
 for t in range(T):
  im=Image.fromarray(base.copy());d=ImageDraw.Draw(im)
  # Family-dependent wall geometry changes with object family, not label.
  yy=57 if 'development' in spec['family'] else 61
  d.line((0,yy,W,yy),fill=(70,70,72),width=2)
  for src,az in enumerate(spec['azimuths']):
   x=W/2-(W/2)/np.tan(np.deg2rad(65))*np.tan(np.deg2rad(az));y=spec['y_centers'][src];r=spec['radii'][src]
   d.line((x,y+r,x,y+r+15),fill=(95,95,98),width=3)
   d.ellipse((x-r-1,y-r-1,x+r+1,y+r+1),outline=(115,115,118),width=1)
   drive=(src+assignment)%2;theta=angle[drive,t];n=spec['blade_count'];color=tuple(spec['colors'][src])
   for k in range(n):
    a=theta+2*np.pi*k/n;half=.19 if 'asymmetric' not in spec['family'] else .13+.035*k
    poly=[(x+rr*np.cos(aa),y+rr*np.sin(aa)) for rr,aa in [(1,a-.3),(r*.88,a-half),(r,a+half),(1,a+.3)]]
    d.polygon(poly,fill=color)
   d.ellipse((x-2,y-2,x+2,y+2),fill=(160,160,165))
  if spec['stratum']=='occluded':
   # Fixed opaque occluder; includes camera-geometry failure cases.
   d.rectangle((18,29,43,39),fill=(55,58,62))
  a=np.asarray(im).astype(float)+rng.normal(0,.45,(H,W,3))
  frames.append(np.clip(np.rint(a),0,255).astype(np.uint8))
 return np.asarray(frames)

def render_audio(spec,assignment):
 rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],spec['index'],809]))
 nt=round(T/FPS*FS);time_axis=np.arange(nt)/FS;env=np.asarray(spec['envelopes'])
 waves=[]
 for event in range(2):
  envelope=np.interp(time_axis,np.arange(T)/FPS,env[event])
  freqs=np.fft.rfftfreq(nt,1/FS);white=rng.normal(size=nt);ft=np.fft.rfft(white)
  center=spec['frequency_centers'][event];mask=np.exp(-.5*((freqs-center)/130)**2)
  dry=np.fft.irfft(ft*mask,n=nt);dry/=max(np.std(dry),1e-12)
  # The rotation control changes amplitude and a small tonal frequency term.
  phase=np.cumsum((center+25*envelope)/FS)*2*np.pi
  dry=(.75*dry+.25*np.sin(phase))*envelope
  waves.append(dry)
 a=np.zeros((nt,4))
 for src,az in enumerate(spec['azimuths']):
  event=(src+assignment)%2;d=np.array([1,np.sin(np.deg2rad(az)),0,np.cos(np.deg2rad(az))])
  a+=np.asarray(waves[event])[:,None]*d[None,:]
  reflected=az+(-1 if src==0 else 1)*55;rd=np.array([1,np.sin(np.deg2rad(reflected)),0,np.cos(np.deg2rad(reflected))])
  delay=spec['reflection_delay'];a[delay:]+=spec['reflection_gain']*waves[event][:-delay,None]*rd[None,:]
 # Diffuse sensor noise is identical for the two owner assignments.
 a+=rng.normal(size=a.shape)*10**(-spec['snr_db']/20)
 return a

def visual_activity(frames):
 # No oracle boxes, object IDs, positions, schedules or poses.
 gray=frames.astype(float).mean(axis=-1)/255
 diff=np.sqrt(np.mean(np.diff(gray,axis=0)**2,axis=1))
 diff=np.vstack([diff[0],diff])
 x=np.arange(W);theta=np.rad2deg(np.arctan((W/2-x)*np.tan(np.deg2rad(65))/(W/2)))
 kernels=np.exp(-.5*((theta[:,None]-AZ[None,:])/5)**2);kernels/=kernels.sum(axis=1,keepdims=True)
 return diff@kernels

def acoustic_activity(a,factorized=True):
 # STFT active intensity from decoded waveform only, W,Y,Z,X convention.
 nfft=512;hop=128;starts=np.arange(0,len(a)-nfft+1,hop)
 ft=np.fft.rfft(np.asarray([a[s:s+nfft] for s in starts])*np.hanning(nfft)[None,:,None],axis=1)
 freqs=np.fft.rfftfreq(nfft,1/FS);ft=ft[:,(freqs>=250)&(freqs<=4200),:]
 cross=np.real(ft[:,:,0].conj()[:,:,None]*ft[:,:,[3,1]])
 energy=np.abs(ft[:,:,0])**2
 if not factorized:cross=cross.sum(axis=1,keepdims=True);energy=energy.sum(axis=1,keepdims=True)
 az=np.rad2deg(np.arctan2(cross[:,:,1],cross[:,:,0]));reliable=np.linalg.norm(cross,axis=-1)/np.maximum(energy,1e-12)
 kernel=np.exp(-.5*((az[:,:,None]-AZ[None,None,:])/6)**2)
 amap=np.sum(kernel*np.sqrt(energy)[:,:,None]*np.clip(reliable[:,:,None],0,1),axis=1)
 stft_t=(starts+nfft/2)/FS
 target=np.arange(T)/FPS
 out=np.stack([np.interp(target,stft_t,amap[:,b]) for b in range(B)],axis=1)
 return out

def normalize_map(m):
 # Subtract temporal floor per direction; cosine over activity, not signed events.
 m=np.maximum(m-np.quantile(m,.15,axis=0,keepdims=True),0)
 return m/max(float(np.linalg.norm(m)),1e-12)

def score_features(v,a_tf,a_bb):
 # The video motion lags at most one frame because temporal differences are causal.
 # Fixed three-frame smoothing is applied identically, not selected per pair.
 def smooth(x):return (np.roll(x,1,axis=0)+x+np.roll(x,-1,axis=0))/3
 v=smooth(v);a_tf=smooth(a_tf);a_bb=smooth(a_bb)
 vn=normalize_map(v);an=normalize_map(a_tf);bn=normalize_map(a_bb)
 return {'temporal':corr(v.sum(axis=1),a_tf.sum(axis=1)),
  'pooled_spatial':corr(v.sum(axis=0),a_tf.sum(axis=0)),
  'broadband_joint':float(np.sum(vn*bn)),
  'factorized_joint':float(np.sum(vn*an))}

def score_packet(frames,a):
 return score_features(visual_activity(frames),acoustic_activity(a,True),acoustic_activity(a,False))

def ba(y,p):
 y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
 return float((np.mean(p[y])+np.mean(~p[~y]))/2)

def thresholds(rows):
 out={}
 for method in METHODS:
  scores=np.asarray([r['scores'][method] for r in rows]);y=np.asarray([r['label'] for r in rows]);u=np.unique(scores)
  cuts=np.r_[u[0]-1,(u[:-1]+u[1:])/2,u[-1]+1];values=[ba(y,scores>=c) for c in cuts]
  # Median maximizer is deterministic and avoids test-driven threshold choice.
  best=np.where(np.asarray(values)>=max(values)-1e-12)[0];k=best[len(best)//2]
  out[method]={'threshold':float(cuts[k]),'development_balanced_accuracy':values[k]}
 return out

def report(rows,thresholds_,bootstrap=False):
 out={};y=np.asarray([r['label'] for r in rows]);scenes=sorted(set(r['scene_id'] for r in rows))
 for method in METHODS:
  pred=np.asarray([r['scores'][method]>=thresholds_[method]['threshold'] for r in rows]);out[method]={'balanced_accuracy':ba(y,pred)}
  for key in ['stratum','family']:
   out[method][key]={value:ba(y[ix],pred[ix]) for value in sorted(set(r[key] for r in rows)) if (ix:=np.asarray([r[key]==value for r in rows])).any()}
  # Ranking within crossed pairs does not depend on the calibrated threshold.
  margins=[]
  for scene in scenes:
   rr=[r for r in rows if r['scene_id']==scene]
   for video in [0,1]:
    pos=next(r for r in rr if r['video_assignment']==video and r['label']==1)
    neg=next(r for r in rr if r['video_assignment']==video and r['label']==0)
    margins.append(pos['scores'][method]-neg['scores'][method])
  out[method]['paired_ranking_accuracy']=float(np.mean(np.asarray(margins)>0)+.5*np.mean(np.asarray(margins)==0))
 if bootstrap:
  rng=np.random.default_rng(90261201);rep={m:[] for m in METHODS};delta=[]
  by_scene={s:[r for r in rows if r['scene_id']==s] for s in scenes}
  for _ in range(2000):
   sample=[r for s in rng.choice(scenes,len(scenes),replace=True) for r in by_scene[s]];yy=[r['label'] for r in sample];v={}
   for m in METHODS:
    v[m]=ba(yy,[r['scores'][m]>=thresholds_[m]['threshold'] for r in sample]);rep[m].append(v[m])
   delta.append(v['factorized_joint']-v['broadband_joint'])
  for m in METHODS:out[m]['scene_bootstrap_95_percentile']=np.quantile(rep[m],[.025,.975]).tolist()
  out['factorized_minus_broadband_95_percentile']=np.quantile(delta,[.025,.975]).tolist()
 return out

def generate(out,stage):
 protocol=read(ROOT/'work/repair09_20261101/protocol.json');split=protocol['split'][stage]
 if stage=='confirmation':
  frozen=read(out/'FROZEN.json')
  assert frozen['code_sha256']==sha(Path(__file__)) and frozen['protocol_sha256']==sha(ROOT/'work/repair09_20261101/protocol.json')
  assert frozen['thresholds_sha256']==sha(out/'thresholds.json')
  assert not (out/'confirmation/result.json').exists(), 'Preserve previous confirmation; do not overwrite.'
 dest=out/stage;dest.mkdir(parents=True,exist_ok=True);start=time.time();rows=[];states=[];marginal=[];shuffled_features=[]
 for i in range(split['scenes']):
  family=split['families'][(i//4)%len(split['families'])];spec=spec_scene(split['seed'],i,family);scene=f'{stage}-{i:03}';folder=dest/'media'/scene;folder.mkdir(parents=True,exist_ok=True)
  frames=[render_video(spec,j) for j in [0,1]];rawaudio=[render_audio(spec,j) for j in [0,1]]
  gain=.8/max(float(np.max(np.abs(a))) for a in rawaudio)
  audio=[];vh=[];ah=[]
  for j in [0,1]:
   vp=folder/f'video{j}.npz';np.savez_compressed(vp,frames=frames[j]);wp=folder/f'audio{j}.wav';wav_write(wp,rawaudio[j]*gain)
   # Read persisted media, not rendering metadata.
   frames[j]=np.load(vp)['frames'];audio.append(wav_read(wp));vh.append(sha(frames[j].tobytes()));ah.append(sha(wp))
  vf=[visual_activity(f) for f in frames];af=[acoustic_activity(a,True) for a in audio];bf=[acoustic_activity(a,False) for a in audio]
  shuffled_features.append((scene,family,spec['stratum'],vf,af,bf))
  for v in [0,1]:
   for a in [0,1]:
    row={'scene_id':scene,'index':i,'family':family,'stratum':spec['stratum'],'video_assignment':v,'audio_assignment':a,'label':int(v==a),'video_hash':vh[v],'audio_hash':ah[a],'scores':score_features(vf[v],af[a],bf[a])};rows.append(row)
  states.append(spec|{'scene_id':scene,'common_audio_gain':gain,'video_hashes':vh,'audio_hashes':ah})
  marginal.append({'scene_id':scene,'video_exact_marginal_equal':all(sum(r['label']==y and r['video_hash']==h for r in rows if r['scene_id']==scene)==1 for h in vh for y in [0,1]),'audio_exact_marginal_equal':all(sum(r['label']==y and r['audio_hash']==h for r in rows if r['scene_id']==scene)==1 for h in ah for y in [0,1])})
  print(f'{stage} {i+1}/{split["scenes"]}',flush=True)
 write(dest/'render_states.json',states);write(dest/'scores.json',rows);write(dest/'marginal_audit.json',marginal)
 if stage=='development':th=thresholds(rows);write(out/'thresholds.json',th)
 else:th=read(out/'thresholds.json')
 result={'stage':stage,'scenes':len(states),'pairs':len(rows),'methods':report(rows,th,stage=='confirmation'),'all_exact_unimodal_marginals_equal':all(r['video_exact_marginal_equal'] and r['audio_exact_marginal_equal'] for r in marginal),
  'audio_only_or_video_only_balanced_accuracy_exact_construction':.5,'wall_seconds':time.time()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'code_sha256':sha(Path(__file__)),'thresholds_sha256':sha(out/'thresholds.json'),'scope':'Controlled procedural feasibility. No learned physics, real transfer, perceptual quality or actual 48GB-machine certification.'}
 # Cross-scene shuffle deliberately breaks both observed owner assignments.
 shuffled=[]
 for ix,(scene,family,stratum,vf,af,bf) in enumerate(shuffled_features):
  candidates=[j for j,item in enumerate(shuffled_features) if item[1]==family and j!=ix];partner=candidates[0] if ix==candidates[-1] else next((j for j in candidates if j>ix),candidates[0]);paf=shuffled_features[partner][4];pbf=shuffled_features[partner][5]
  for v in [0,1]:
   for a in [0,1]:shuffled.append({'scene_id':scene,'family':family,'stratum':stratum,'video_assignment':v,'audio_assignment':a,'label':int(v==a),'scores':score_features(vf[v],paf[a],pbf[a]),'audio_partner_scene':shuffled_features[partner][0]})
 write(dest/'shuffled_scores.json',shuffled);result['cross_scene_shuffled_methods']=report(shuffled,th,False)
 write(dest/'result.json',result);print(json.dumps(result,indent=2))

def freeze(out):
 assert (out/'development/result.json').exists();assert not (out/'FROZEN.json').exists()
 write(out/'FROZEN.json',{'status':'frozen_before_confirmation','code_sha256':sha(Path(__file__)),'protocol_sha256':sha(ROOT/'work/repair09_20261101/protocol.json'),'development_result_sha256':sha(out/'development/result.json'),'thresholds_sha256':sha(out/'thresholds.json'),'utc_unix':time.time()})

def example(out,stage,index):
 folder=out/stage/'media'/f'{stage}-{index:03}';frames=np.load(folder/'video0.npz')['frames'];dest=folder/'coherent_example.mp4'
 p=subprocess.run(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(folder/'audio0.wav'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-ac','2','-shortest',str(dest)],input=frames.tobytes(),capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode())
 Image.fromarray(frames[30]).resize((512,288)).save(folder/'preview.png')
 print(dest)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','freeze','confirmation','example']);p.add_argument('--out',type=Path,required=True);p.add_argument('--example-stage',default='confirmation');p.add_argument('--index',type=int,default=0);args=p.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 if args.stage=='freeze':freeze(args.out)
 elif args.stage=='example':example(args.out,args.example_stage,args.index)
 else:
  write(args.out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'hardware':subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True).stdout.strip(),'cpu_only':True,'api_calls':0})
  generate(args.out,args.stage)
if __name__=='__main__':main()
