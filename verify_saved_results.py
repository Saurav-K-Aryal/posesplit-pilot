"""Recompute EventOwner metrics, marginals, freeze hashes and sample media."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,importlib.util,json,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/repair09-eventowner-v2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):
    if not np.allclose(a,b,atol=1e-12,rtol=1e-12):raise AssertionError((a,b))
spec=importlib.util.spec_from_file_location('event_owner',ROOT/'pilots/repair09/event_owner.py')
eo=importlib.util.module_from_spec(spec);spec.loader.exec_module(eo)

def verify():
    frozen=read(RUN/'FROZEN.json')
    assert sha(ROOT/'pilots/repair09/event_owner.py')==frozen['code_sha256']
    assert sha(ROOT/'work/repair09_20261101/protocol.json')==frozen['protocol_sha256']
    assert sha(RUN/'thresholds.json')==frozen['thresholds_sha256']
    assert sha(RUN/'development/result.json')==frozen['development_result_sha256']
    rows=read(RUN/'confirmation/scores.json');expected=read(RUN/'confirmation/result.json');thresholds=read(RUN/'thresholds.json')
    grouped=defaultdict(list)
    for r in rows:grouped[r['scene_id']].append(r)
    scenes=sorted(grouped)
    assert len(rows)==480 and len(scenes)==120
    for scene,rr in grouped.items():
        assert len(rr)==4 and Counter(r['label'] for r in rr)=={0:2,1:2}
        for key in ['video_hash','audio_hash']:
            assert len(set(r[key] for r in rr))==2
            for value in set(r[key] for r in rr):assert Counter(r['label'] for r in rr if r[key]==value)=={0:1,1:1}
        for r in rr:assert r['label']==int(r['video_assignment']==r['audio_assignment'])
    ix=np.random.default_rng(90261201).integers(0,len(scenes),size=(2000,len(scenes)))
    clusters={};results={}
    for method in eo.METHODS:
        th=thresholds[method]['threshold']
        values=[]
        for scene in scenes:
            rr=grouped[scene];y=np.array([r['label'] for r in rr],bool);pred=np.array([r['scores'][method]>=th for r in rr])
            values.append((pred[y].mean()+(~pred[~y]).mean())/2)
        clusters[method]=np.array(values);ba=float(np.mean(values));close(ba,expected['methods'][method]['balanced_accuracy'])
        ci=np.quantile(clusters[method][ix].mean(1),[.025,.975]);close(ci,expected['methods'][method]['scene_bootstrap_95_percentile'])
        margins=[]
        for rr in grouped.values():
            for video in [0,1]:
                margins.append(next(r['scores'][method] for r in rr if r['video_assignment']==video and r['label']==1)-next(r['scores'][method] for r in rr if r['video_assignment']==video and r['label']==0))
        rank=float(np.mean(np.array(margins)>0)+.5*np.mean(np.array(margins)==0));close(rank,expected['methods'][method]['paired_ranking_accuracy'])
        for stratum in set(r['stratum'] for r in rows):
            vv=[values[i] for i,s in enumerate(scenes) if grouped[s][0]['stratum']==stratum];close(np.mean(vv),expected['methods'][method]['stratum'][stratum])
        results[method]={'balanced_accuracy':ba,'scene_95_percentile':ci.tolist(),'paired_ranking_accuracy':rank}
    delta=np.quantile((clusters['factorized_joint']-clusters['broadband_joint'])[ix].mean(1),[.025,.975]);close(delta,expected['methods']['factorized_minus_broadband_95_percentile'])
    shuffled=read(RUN/'confirmation/shuffled_scores.json')
    for method in eo.METHODS:
        y=np.array([r['label'] for r in shuffled],bool);pred=np.array([r['scores'][method]>=thresholds[method]['threshold'] for r in shuffled])
        close((pred[y].mean()+(~pred[~y]).mean())/2,expected['cross_scene_shuffled_methods'][method]['balanced_accuracy'])
    # Regenerate exactly the two released lossless scenes using stored render state.
    states={r['scene_id']:r for r in read(RUN/'confirmation/render_states.json')};sample_checks=[]
    for scene in ['confirmation-014','confirmation-035']:
        state=states[scene];folder=ROOT/'examples/lossless'/scene
        for assignment in [0,1]:
            frames=eo.render_video(state,assignment);stored=np.load(folder/f'video{assignment}.npz')['frames'];assert np.array_equal(frames,stored)
            with tempfile.TemporaryDirectory() as temp:
                wp=Path(temp)/'audio.wav';eo.wav_write(wp,eo.render_audio(state,assignment)*state['common_audio_gain']);assert sha(wp)==sha(folder/f'audio{assignment}.wav')
        for r in grouped[scene]:
            v=np.load(folder/f"video{r['video_assignment']}.npz")['frames'];a=eo.wav_read(folder/f"audio{r['audio_assignment']}.wav");scores=eo.score_packet(v,a)
            for m in eo.METHODS:close(scores[m],r['scores'][m])
        sample_checks.append(scene)
    # Real result replay is numerical only; source recordings are not included.
    real=read(RUN/'real_component/scores.json');real_expected=read(RUN/'real_component/result.json');rng=np.random.default_rng(90261211)
    manifest=read(ROOT/'work/repair09_20261101/real_source_manifest.json');assert sha(ROOT/'work/repair09_20261101/real_source_manifest.json')==read(RUN/'real_source_manifest_freeze.json')['sha256']
    assert len(real)==144 and len(manifest['records'])==9
    for method in eo.METHODS:
        by_recording=[]
        for rec in manifest['records']:
            rr=[r for r in real if r['recording']==rec['recording']];win=[]
            for sec in rec['start_seconds']:
                positive=next(r['scores'][method] for r in rr if r['seconds']==sec and r['label']==1);negative=next(r['scores'][method] for r in rr if r['seconds']==sec and r['label']==0);win.append(positive>negative)
            by_recording.append(np.mean(win))
        close(np.mean(by_recording),real_expected['methods'][method]['matched_pair_ranking_accuracy'])
        draws=rng.choice(by_recording,size=(2000,9),replace=True).mean(1);close(np.quantile(draws,[.025,.975]),real_expected['methods'][method]['recording_cluster_bootstrap_95_percentile'])
    # Independently recompute the tiny linear counterexample without writing files.
    rng=np.random.default_rng(90261127);a=rng.normal(size=(32,96));b=rng.normal(size=(32,96));changed=a+rng.normal(0,.7,a.shape);za=rng.normal(0,.1,a.shape);zb=rng.normal(0,.1,b.shape)
    def loss(weight):return np.mean((weight*a+.9*b+zb-(weight*changed+.9*b+zb))**2)
    analytic=2*(-.17)*np.mean((changed-a)**2);finite=(loss(-.17+1e-6)-loss(-.17-1e-6))/(2e-6)
    check=read(ROOT/'audits/eventowner/loss_nonvacuity_check.json');close(loss(-.17),check['unchanged_b_loss']);close(analytic,check['analytic_gradient_wrt_A_to_B_weight']);assert abs(finite-analytic)<1e-8 and loss(0)==0
    result={'status':'passed','scenes':120,'pairs':480,'all_exact_marginals_balanced':True,'unimodal_balanced_accuracy_by_construction':.5,'metrics':results,'paired_delta_interval':delta.tolist(),'sample_media_regenerated':sample_checks,'real_numeric_windows':72,'real_raw_sources_included':False,'frozen_hashes_match':True,'nonvacuity_counterexample_checked':True}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':verify()
