"""Exploratory real-pixel/FOA component on predeclared, previously exposed STARSS23.

No metadata directions, class annotations, boxes or source IDs enter scoring.
This is not confirmation of procedural realism or causal ownership on real data.
"""
from pathlib import Path
import hashlib,json,subprocess,time,resource
import numpy as np
import event_owner as eo

ROOT=Path(__file__).resolve().parents[2]
FFMPEG='/opt/homebrew/bin/ffmpeg'
H=128;W=256;T=96;B=48;AZ=np.linspace(-180,180,B,endpoint=False)+180/B

def run(args):
 p=subprocess.run([FFMPEG,'-hide_banner','-loglevel','error',*args],capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr.decode())
 return p.stdout

def visual(frames):
 gray=frames.astype(float).mean(axis=-1)/255
 d=np.sqrt(np.mean(np.diff(gray,axis=0)**2,axis=1));d=np.vstack([d[0],d])
 theta=180-360*np.arange(W)/W
 delta=(theta[:,None]-AZ[None,:]+180)%360-180
 k=np.exp(-.5*(delta/10)**2);k/=k.sum(axis=1,keepdims=True)
 return d@k

def acoustic(a,factorized):
 starts=np.arange(0,len(a)-512+1,128)
 ft=np.fft.rfft(np.asarray([a[s:s+512] for s in starts])*np.hanning(512)[None,:,None],axis=1)
 freq=np.fft.rfftfreq(512,1/16000);ft=ft[:,(freq>=250)&(freq<=4200),:]
 iv=np.real(ft[:,:,0].conj()[:,:,None]*ft[:,:,[3,1]]);energy=np.abs(ft[:,:,0])**2
 if not factorized:iv=iv.sum(axis=1,keepdims=True);energy=energy.sum(axis=1,keepdims=True)
 az=np.rad2deg(np.arctan2(iv[:,:,1],iv[:,:,0]));rel=np.clip(np.linalg.norm(iv,axis=-1)/np.maximum(energy,1e-12),0,1)
 delta=(az[:,:,None]-AZ[None,None,:]+180)%360-180
 amap=(np.exp(-.5*(delta/10)**2)*np.sqrt(energy)[:,:,None]*rel[:,:,None]).sum(axis=1)
 ts=(starts+256)/16000;return np.stack([np.interp(np.arange(T)/24,ts,amap[:,b]) for b in range(B)],axis=1)

def main():
 start=time.time();out=ROOT/'runs/repair09-eventowner-v2/real_component';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'result.json').exists(),'Preserve completed exploratory result.'
 manifest_path=ROOT/'work/repair09_20261101/real_source_manifest.json';manifest=eo.read(manifest_path)
 frozen=eo.read(ROOT/'runs/repair09-eventowner-v2/real_source_manifest_freeze.json');assert frozen['sha256']==eo.sha(manifest_path)
 records=manifest['records'];allfeatures=[];decode=[]
 for rec in records:
  ap=ROOT/rec['audio_path'];vp=ROOT/rec['video_path'];assert eo.sha(ap)==rec['audio_sha256'] and eo.sha(vp)==rec['video_sha256']
  # All32 seconds are decoded once. No score-dependent clip selection.
  video=np.frombuffer(run(['-i',str(vp),'-t','32','-vf',f'fps=24,scale={W}:{H}','-f','rawvideo','-pix_fmt','rgb24','pipe:1']),dtype=np.uint8).reshape(-1,H,W,3)
  audio=np.frombuffer(run(['-i',str(ap),'-t','32','-ar','16000','-ac','4','-f','f32le','pipe:1']),dtype='<f4').reshape(-1,4).astype(float)
  assert len(video)==768 and len(audio)==512000
  rows=[]
  for sec in rec['start_seconds']:
   v=video[int(sec*24):int((sec+4)*24)];a=audio[int(sec*16000):int((sec+4)*16000)]
   rows.append({'v':visual(v),'a':acoustic(a,True),'b':acoustic(a,False)})
   decode.append({'recording':rec['recording'],'room':rec['room'],'seconds':sec,'rgb_sha256':eo.sha(v.tobytes()),'float32_pcm_sha256':eo.sha(a.astype('<f4').tobytes()),'rgb_frames':len(v),'audio_samples':len(a)})
  allfeatures.append(rows);print('decoded',rec['recording'],flush=True)
 scores=[]
 for ri,rec in enumerate(records):
  partner=(ri+1)%len(records)
  for wi,sec in enumerate(rec['start_seconds']):
   v=allfeatures[ri][wi]['v']
   for label,ar in [(1,ri),(0,partner)]:
    aa=allfeatures[ar][wi]
    scores.append({'recording':rec['recording'],'room':rec['room'],'seconds':sec,'label':label,'audio_recording':records[ar]['recording'],'scores':eo.score_features(v,aa['a'],aa['b'])})
 result={'status':'exploratory_component_only','recordings':9,'rooms':9,'windows':72,'pairs':144,'previous_room_recording_exposure':True,'source_manifest_sha256':eo.sha(manifest_path),'methods':{},'wall_seconds':time.time()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['No real causal ownership gold labels.','No new untouched real rooms.','Random cross-recording mismatch may change semantics and activity inventory, unlike exact ownership crossings.','No real media source labels or metadata used in scoring.','No human quality study or real transfer claim.']}
 rng=np.random.default_rng(90261211)
 for method in eo.METHODS:
  diffs=[];rooms=[]
  for ri,rec in enumerate(records):
   rr=[r for r in scores if r['recording']==rec['recording']];dd=[next(r['scores'][method] for r in rr if r['seconds']==sec and r['label']==1)-next(r['scores'][method] for r in rr if r['seconds']==sec and r['label']==0) for sec in rec['start_seconds']]
   diffs+=dd;rooms.append(float(np.mean(np.asarray(dd)>0)))
  reps=[float(np.mean(rng.choice(rooms,len(rooms),replace=True))) for _ in range(2000)]
  result['methods'][method]={'matched_pair_ranking_accuracy':float(np.mean(np.asarray(diffs)>0)),'mean_matched_minus_shuffle_score':float(np.mean(diffs)),'recording_cluster_bootstrap_95_percentile':np.quantile(reps,[.025,.975]).tolist(),'by_recording':[{'recording':r['recording'],'accuracy':v} for r,v in zip(records,rooms)]}
 eo.write(out/'decoded_media_audit.json',decode);eo.write(out/'scores.json',scores);eo.write(out/'result.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
