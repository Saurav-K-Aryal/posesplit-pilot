"""Frozen-noise coupled-model counterexample; not a learned synthesis result.

Changing source A's event at fixed identities need not preserve source B in a
coupled residual network. A row-permutation check would not establish this.
"""
from pathlib import Path
import json
import numpy as np

def main():
 rng=np.random.default_rng(90261127);a=rng.normal(size=(32,96));b=rng.normal(size=(32,96));changed=a+rng.normal(0,.7,a.shape)
 # Fixed per-source noise remains identical in the two interventions.
 za=rng.normal(0,.1,a.shape);zb=rng.normal(0,.1,b.shape)
 theta=np.array([.8,.23,-.17,.9])
 def coupled(parameters,a_input):
  aa,ab,ba,bb=parameters
  return aa*a_input+ab*b+za, ba*a_input+bb*b+zb
 def unchanged_b_loss(parameters):
  original=coupled(parameters,a)[1];intervened=coupled(parameters,changed)[1]
  return float(np.mean((original-intervened)**2))
 delta=changed-a;loss=unchanged_b_loss(theta);analytic=2*theta[2]*float(np.mean(delta**2))
 eps=1e-6;plus=theta.copy();minus=theta.copy();plus[2]+=eps;minus[2]-=eps
 finite=(unchanged_b_loss(plus)-unchanged_b_loss(minus))/(2*eps)
 independent=theta.copy();independent[2]=0
 assert loss>0 and abs(analytic)>0 and abs(finite-analytic)<1e-8
 assert unchanged_b_loss(independent)==0
 result={'purpose':'Method non-vacuity sanity check only, not training or perceptual benefit.','seed':90261127,'intervention':'Change only A event drive at fixed physical IDs, B, room and per-source noise.','coupled_model':'Aout=.8*A+.23*B+zA; Bout=-.17*A+.9*B+zB','unchanged_b_loss':loss,'analytic_gradient_wrt_A_to_B_weight':analytic,'finite_difference_gradient':finite,'absolute_gradient_error':abs(finite-analytic),'hard_independent_source_loss':unchanged_b_loss(independent),'no_training_performed':True,'limitation':'Linear counterexample proves proposed constraint can be nonzero, not that the eventual spectral generator benefits or that consistency alone prevents constant outputs.'}
 out=Path(__file__).resolve().parents[2]/'work/repair09_20261101/loss_nonvacuity_check.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
