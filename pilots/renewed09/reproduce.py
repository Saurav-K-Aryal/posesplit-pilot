"""Run an authorized fresh PoseSplit reproduction using reviewed source bindings."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,shutil,subprocess,sys,time,zlib
ROOT=Path(__file__).resolve().parents[2]
EXPECTED_CODE='bcd90a51205349cf71007ea4f8b86181888ee3148c89aed385dc8f14099f939d'
EXPECTED_PROTOCOL='5b977a500959d4e5aeda2d81f5f0fbc13a307de642bfa0167f1129aeecaf6522'
EXPECTED_SELECTION='38ba020d654e427a7cd29450c566f0f92a9afbdfccbb31924f5ff87f87f6ff4e'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cache',type=Path,default=ROOT/'runs/renewed-09-posesplit-20261031');args=parser.parse_args();out=args.out.resolve();cache=args.cache.resolve()
 if out.exists():raise FileExistsError('Fresh reproduction requires a new output directory; observed runs are never overwritten.')
 source=ROOT/'pilots/renewed09/pose_split.py';reviewed=ROOT/'pilots/renewed09/reviewed_protocol.json';assert sha(source)==EXPECTED_CODE,'Reviewed source changed; review a new version before reproducing.';assert sha(reviewed)==EXPECTED_PROTOCOL
 metadata=ROOT/'work/task09_design_sources/metadata_dev.zip';spec=load(reviewed);metadata.parent.mkdir(parents=True,exist_ok=True)
 if not metadata.exists():
  import urllib.request
  with urllib.request.urlopen('https://zenodo.org/api/records/7880637/files/metadata_dev.zip/content',timeout=60) as response:data=response.read()
  if hashlib.md5(data).hexdigest()!=spec['sources']['metadata_md5']:raise RuntimeError('Official metadata checksum differs from reviewed source.')
  metadata.write_bytes(data)
 assert hashlib.md5(metadata.read_bytes()).hexdigest()==spec['sources']['metadata_md5']
 protocol=ROOT/'work/task09_protocol.json';protocol.parent.mkdir(parents=True,exist_ok=True)
 if protocol.exists() and sha(protocol)!=EXPECTED_PROTOCOL:
  backup=protocol.with_name('task09_protocol.before-replay-'+str(time.time_ns())+'.json');shutil.copy2(protocol,backup);print('Preserved previous working protocol at',backup,flush=True)
 shutil.copy2(reviewed,protocol)
 def stage(name):
  print('\nPoseSplit:',name,flush=True)
  assert sha(source)==EXPECTED_CODE and sha(protocol)==EXPECTED_PROTOCOL
  subprocess.run([sys.executable,'-m','pilots.renewed09.pose_split','--stage',name,'--out',str(out)],cwd=ROOT,check=True)
 stage('plan');assert sha(out/'selection.json')==EXPECTED_SELECTION,'Labels-only selected cohort differs; stop without decoding media.'
 versions={}
 for name in ['numpy','Pillow','fsspec','aiohttp']:
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]='missing'
 write(out/'reproduction_environment.json',{'python':sys.version,'platform':platform.platform(),'dependencies':versions,'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],'reproduction_driver_sha256':sha(Path(__file__)),'same_reviewed_source_protocol_and_selection':True})
 stage('freeze')
 # Reuse only ordinary hard copies whose SHA and immutable member CRC/length match.
 reused=[];readback=cache/'source_readback.json'
 if cache!=out and readback.is_file() and load(readback).get('status')=='passed':
  saved={r['member']:r for r in load(readback)['files']}
  for row in load(out/'selection.json'):
   for kind in ['audio','video']:
    member=row[kind+'_member'];info=row[kind+'_catalog'];path=cache/'sources'/member
    if member not in saved or not path.is_file() or path.is_symlink():continue
    raw=path.read_bytes()
    if len(raw)!=info['bytes'] or zlib.crc32(raw)!=info['crc32'] or hashlib.sha256(raw).hexdigest()!=saved[member]['sha256']:raise RuntimeError('Cache checksum mismatch: '+member)
    target=out/'sources'/member;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);reused.append({'member':member,'sha256':sha(target),'bytes':len(raw)})
 write(out/'source_cache_reuse.json',{'ordinary_file_copies':reused,'files_reused':len(reused),'symlinks_created':False,'source_cache':str(cache),'policy':'Only SHA/CRC/size-verified files reused; official selected-source fetch validates every member again.'})
 stage('fetch');stage('run')
 write(out/'reproduction_readback.json',{'status':'completed','code_sha256':sha(source),'protocol_sha256':sha(out/'protocol.json'),'selection_sha256':sha(out/'selection.json'),'result_sha256':sha(out/'result.json'),'source_cache_files_reused':len(reused),'native_model_calls':0,'paid_cloud_calls':0})
 print('\nFinished. Results:',out/'result.json',flush=True)
if __name__=='__main__':main()
