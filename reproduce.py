"""Portable entry point around the unchanged frozen EventOwner renderer."""
from pathlib import Path
import argparse, importlib.util, json, platform, sys

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('event_owner',ROOT/'pilots/repair09/event_owner.py')
eo=importlib.util.module_from_spec(spec);spec.loader.exec_module(eo)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists():p.error('Choose a new output directory; existing runs are preserved.')
    out.mkdir(parents=True)
    eo.write(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':eo.np.__version__,'cpu_only':True,'api_calls':0,'entrypoint':'reproduce.py; unchanged frozen feature/generation functions'})
    eo.generate(out,'development');eo.freeze(out);eo.generate(out,'confirmation')
    observed=eo.read(out/'confirmation/result.json')
    expected=eo.read(ROOT/'runs/repair09-eventowner-v2/confirmation/result.json')
    # Timings and host records differ; exact media and scores must reproduce.
    for name in ['scores.json','shuffled_scores.json','marginal_audit.json','render_states.json']:
        assert eo.read(out/'confirmation'/name)==eo.read(ROOT/'runs/repair09-eventowner-v2/confirmation'/name),name
    assert observed['methods']==expected['methods']
    eo.write(out/'reproduction_check.json',{'all_confirmation_arrays_and_metrics_equal':True,'scenes':observed['scenes'],'pairs':observed['pairs'],'original_results_preserved':True})
    print('Exact confirmation scores, hashes, scene state and metrics reproduced:',out)

if __name__=='__main__':main()
