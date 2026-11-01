"""Rebuild the current EventOwner numerical figures from saved results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def main():
    data=json.loads((ROOT/'runs/repair09-eventowner-v2/confirmation/result.json').read_text())
    methods=['temporal','pooled_spatial','broadband_joint','factorized_joint'];names=['Temporal','Pooled spatial','Broadband joint','Frequency resolved']
    fig,ax=plt.subplots(1,2,figsize=(11.3,3.8),layout='constrained')
    y=[data['methods'][m]['balanced_accuracy'] for m in methods];ci=np.array([data['methods'][m]['scene_bootstrap_95_percentile'] for m in methods]);colors=['#969696','#737373','#4a78a4','#1b4e7a']
    ax[0].barh(names,y,color=colors,xerr=np.stack([np.array(y)-ci[:,0],ci[:,1]-np.array(y)]),capsize=3)
    ax[0].axvline(.5,color='black',ls='--',lw=.8);ax[0].set_xlim(0,1.02);ax[0].set_xlabel('Balanced accuracy');ax[0].set_title('All 120 confirmation scenes')
    strata=['separated','occluded','near_colocated','overlap'];x=np.arange(4);w=.35
    for off,m,col,label in [(-w/2,'broadband_joint',colors[2],'Broadband joint'),(w/2,'factorized_joint',colors[3],'Frequency resolved')]:
        ax[1].bar(x+off,[data['methods'][m]['stratum'][s] for s in strata],w,color=col,label=label)
    ax[1].set_xticks(x,['Separated','Toy occlusion','Near colocated','Overlap'],rotation=15);ax[1].axhline(.5,color='black',ls='--',lw=.8);ax[1].set_ylim(0,1.09);ax[1].set_ylabel('Balanced accuracy');ax[1].set_title('Stress strata (30 scenes each)');ax[1].legend(loc='lower left',fontsize=8)
    for a in ax:a.spines[['top','right']].set_visible(False)
    folder=ROOT/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'confirmation_results.png',dpi=220);plt.close(fig)
    real=json.loads((ROOT/'runs/repair09-eventowner-v2/real_component/result.json').read_text());fig,a=plt.subplots(figsize=(6.5,3.2),layout='constrained')
    y=[real['methods'][m]['matched_pair_ranking_accuracy'] for m in methods];ci=np.array([real['methods'][m]['recording_cluster_bootstrap_95_percentile'] for m in methods])
    a.barh(names,y,color=colors,xerr=np.stack([np.array(y)-ci[:,0],ci[:,1]-np.array(y)]),capsize=3);a.axvline(.5,color='black',ls='--',lw=.8);a.set_xlim(0,1);a.set_xlabel('Matched versus shuffled ranking accuracy');a.set_title('Exploratory real component (9 exposed rooms)');a.spines[['top','right']].set_visible(False);fig.savefig(folder/'real_component.png',dpi=220);plt.close(fig)
    (ROOT/'figure_data.json').write_text(json.dumps({'confirmation':data,'real':real,'interpretation':'Synthetic confidence intervals are scene clustered. Real intervals are recording/room clustered; real result is not causal-ownership ground truth.'},indent=2)+'\n')
if __name__=='__main__':main()
