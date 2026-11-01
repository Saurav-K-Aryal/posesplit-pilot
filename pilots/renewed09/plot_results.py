"""Plot completed PoseSplit evidence; no data decoding, fitting or model calls."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=ROOT/'runs/renewed-09-posesplit-20261031');p.add_argument('--out',type=Path);a=p.parse_args();study=a.study.resolve();out=(a.out or study/'figures').resolve();out.mkdir(parents=True,exist_ok=True)
 result=load(study/'result.json');assert result['status']=='completed' and result['model_calls']==0
 for name,digest in result['source_hashes'].items():assert sha(study/name)==digest
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 arms=['analytic','pose_aware','canonical_matched_orientation','wrong_pose_negative_control'];labels=['Analytic intensity\n+ known pose','Pose-aware\ncurriculum','Canonical matched\norientation','Wrong-pose labels\nnegative control'];colors=['#009e73','#4c78a8','#e69f00','#999999']
 rows={r['arm']:r for r in result['view_metrics'] if r['regime']=='camera_only'};assert set(rows)==set(arms)
 fig,ax=plt.subplots(figsize=(8.5,3.65),layout='constrained')
 for i,(arm,label,color) in enumerate(zip(arms,labels,colors)):
  row=rows[arm];value=row['primary_mean_seed_pooled_balanced_accuracy'];ax.bar(i,value,color=color,width=.64)
  ci=row['room_cluster_bootstrap']['confidence95']
  if ci is not None:ax.vlines(i,ci[0],ci[1],color='black',linewidth=1.2);ax.hlines(ci,[i-.08,i-.08],[i+.08,i+.08],color='black',linewidth=1.2)
  ax.text(i,max(value,ci[1] if ci else value)+.025,f'{value:.3f}',ha='center',fontsize=10)
 ax.axhline(.5,color='#666666',linestyle=':',linewidth=.8,label='Balanced-accuracy reference');ax.set_ylim(0,1.15);ax.set_yticks(np.arange(0,1.01,.2));ax.set_xticks(range(4),labels,fontsize=9);ax.set_ylabel('Mean seed pooled balanced accuracy');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='upper right',frameon=False,fontsize=8)
 ax.set_title(f"Real camera-only views: {result['locked_windows']} windows, 14 recordings, 7 rooms",fontsize=11)
 fig.savefig(out/'pose_availability.png',dpi=220,bbox_inches='tight');plt.close(fig)
 real=[r for r in load(study/'real_features.json') if r['split']=='locked' and r['regime']=='camera_only' and r['heading']==0];assert len(real)==result['locked_windows']
 errors=np.array([r['audio_angular_error_deg'] for r in real]);front=np.array([np.rad2deg(np.arccos(np.clip(r['gold_unit_direction'][0],-1,1))) for r in real]);assert abs(np.median(errors)-result['analytic_audio_real_angular_error']['median_deg'])<1e-10;assert abs(np.median(front)-result['constant_front_real_angular_error']['median_deg'])<1e-10
 fig,ax=plt.subplots(figsize=(8.5,3.3),layout='constrained')
 for values,label,color in [(errors,'Analytic intensity','#009e73'),(front,'Constant front','#b44e3a')]:
  x=np.sort(values);y=np.arange(1,len(x)+1)/len(x);ax.step(np.r_[0,x],np.r_[0,y],where='post',label=f'{label} (median {np.median(x):.1f}°)',color=color,linewidth=1.8)
 ax.axvline(20,linestyle=':',color='#888888',linewidth=1);ax.set_xlim(0,180);ax.set_ylim(0,1.03);ax.set_xlabel('3D angular error relative to motion-capture direction (degrees)');ax.set_ylabel('Fraction of held-out windows');ax.set_title('Real audio localization before camera-pose projection',fontsize=11);ax.grid(alpha=.18);ax.legend(loc='lower right',frameon=False,fontsize=9)
 fig.savefig(out/'real_doa_ecdf.png',dpi=220,bbox_inches='tight');plt.close(fig)
 for split in ['development','locked']:
  source=study/'media_audit'/f'montage_{split}.png'
  if source.is_file():shutil.copy2(source,out/f'media_montage_{split}.png')
 manifest={'status':'generated_from_completed_observed_results','study':str(study.relative_to(ROOT)) if study.is_relative_to(ROOT) else str(study),'source_result_sha256':sha(study/'result.json'),'source_real_features_sha256':sha(study/'real_features.json'),'source_media_audit_sha256':sha(study/'media_audit.json'),'generator_sha256':sha(Path(__file__)),'figures':{p.name:sha(p) for p in sorted(out.glob('*.png'))},'figure_notes':{'pose_availability':'Camera-onlyprimarypooledBAaveragedacross5seeds;analyticbaselineonescore;intervalsresample7roomclustersandarenotindependent-windowuncertainty. Wronglabelsareanintentionallyinvalidnegativecontrol.','real_doa_ecdf':'DescriptiveECDFof168nestedwindows;sourceannotationsareexternalMoCaplabels;noindependenceclaimorCI.','montages':'Twofixedlexicographicexamplesdefinedbeforeoutcomes;realpixels,geometryauditonly,noimagefeaturesusedbyhead.'},'native_model_calls':0}
 (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':'passed','figures':list(manifest['figures']),'result_sha256':manifest['source_result_sha256']},indent=2))
if __name__=='__main__':main()
