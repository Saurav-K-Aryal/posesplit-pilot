"""Replay the exploratory real component with separately obtained STARSS23 files."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path,help='Extracted STARSS23 source tree or a completed PoseSplit sources directory; no downloads are performed.')
    p.add_argument('--output',required=True,type=Path,help='A fresh output directory.')
    p.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    args=p.parse_args()
    if not args.source_root.is_dir():p.error('Source root is missing.')
    if args.output.exists():p.error('Choose a new output directory; existing results are preserved.')
    if not args.ffmpeg:p.error('FFmpeg is required; install it or pass --ffmpeg.')
    manifest_path=ROOT/'work/repair09_20261101/real_source_manifest.json';manifest=json.loads(manifest_path.read_text());frozen=json.loads((ROOT/'runs/repair09-eventowner-v2/real_source_manifest_freeze.json').read_text());assert sha(manifest_path)==frozen['sha256']
    with tempfile.TemporaryDirectory(prefix='eventowner-real-') as temp:
        workspace=Path(temp)
        for name in ['event_owner.py','real_component.py']:
            dest=workspace/'pilots/repair09'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'pilots/repair09'/name,dest)
        dest=workspace/'work/repair09_20261101/real_source_manifest.json';dest.parent.mkdir(parents=True);shutil.copy2(manifest_path,dest)
        dest=workspace/'runs/repair09-eventowner-v2/real_source_manifest_freeze.json';dest.parent.mkdir(parents=True);shutil.copy2(ROOT/'runs/repair09-eventowner-v2/real_source_manifest_freeze.json',dest)
        for rec in manifest['records']:
            for kind in ['audio','video']:
                name=Path(rec[kind+'_path']).name;matches=[f for f in args.source_root.rglob(name) if f.is_file() and sha(f)==rec[kind+'_sha256']]
                if not matches:raise FileNotFoundError('Missing or mismatched source: '+name)
                target=workspace/rec[kind+'_path'];target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(matches[0].resolve())
        load('event_owner',workspace/'pilots/repair09/event_owner.py');rc=load('real_component',workspace/'pilots/repair09/real_component.py');rc.FFMPEG=args.ffmpeg;rc.main()
        result=workspace/'runs/repair09-eventowner-v2/real_component';original=ROOT/'runs/repair09-eventowner-v2/real_component'
        assert json.loads((result/'scores.json').read_text())==json.loads((original/'scores.json').read_text())
        assert json.loads((result/'decoded_media_audit.json').read_text())==json.loads((original/'decoded_media_audit.json').read_text())
        shutil.copytree(result,args.output)
    print('Real pixels/PCM hashes and all real scores reproduced:',args.output)
if __name__=='__main__':main()
