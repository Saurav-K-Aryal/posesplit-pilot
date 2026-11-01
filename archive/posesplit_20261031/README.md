# PoseSplit

**PoseSplit: A Synthetic-to-Real Audio-Plus-Pose Pilot for Camera Reframing**  
Saurav K. Aryal, Howard University.

PoseSplit tests whether procedural spatial audio can train a small predictor of geometric camera viewport membership using audio and known camera–microphone pose. Its real evaluation uses 168 half-second windows from 14 STARSS23 recordings in seven held-out rooms. Independent-pose training helps camera-only reframing, while canonical matched-orientation training performs slightly better under co-rig rotation. No image features enter the learned predictors.

## Results

| Training or baseline | Camera-only reframing | Co-rig virtual rotation |
|---|---:|---:|
| Pose-aware procedural curriculum | 0.8294 | 0.8319 |
| Canonical matched orientation | 0.4800 | 0.8409 |
| Wrong-pose labels: diagnostic control | 0.4081 | 0.4179 |
| Analytic intensity + known pose | 0.7817 | 0.7817 |

Values are balanced accuracy, pooled within each seed and averaged across five training seeds. The analytic method has no learned seed. The 168 locked windows produce 672 dependent views per regime and predictor. The pose-aware camera-only gain over the analytic method is 0.0476, with a paired seven-room percentile interval of [0.0067, 0.1038]. These are small-cohort estimates; the co-rig counter-result rules out a claim of universal curriculum superiority.

Analytic localization has a median 3D angular error of 23.48°, versus 50.14° for a constant-front reference. This is separate signal-processing evidence: the learned predictors estimate binary viewport membership, not direction. Wrong-pose labels are an intentionally invalid negative control.

## Replay saved results without recordings

```sh
python3 -m venv .venv-pilots
.venv-pilots/bin/python -m pip install numpy==2.5.4
.venv-pilots/bin/python verify_saved_results.py
```

The verifier replays saved features, the 15 small NumPy checkpoints, all 35,328 development/test predictions, and room-bootstrap statistics. It does not reconstruct omitted audio/video inputs. The full raw-data audit is preserved in `audits/independent_raw_audit_C.json`; source manifests allow a fresh reconstruction.

To regenerate the two numerical figures without photos, model fitting, or media downloads:

```sh
.venv-pilots/bin/python -m pip install -r requirements-figures.txt
.venv-pilots/bin/python -m pilots.renewed09.plot_results --study runs/renewed-09-posesplit-20261031 --out figures
```

## Run a fresh study with one command

Python 3 and FFmpeg must be installed. The launcher creates a new timestamped run and logs, uses `.venv-pilots`, installs missing Python dependencies there, verifies reviewed source/protocol/selection hashes, and executes plan → freeze → fetch → run:

```sh
bash run_task09.command
```

It was tested end to end in the original workspace: every prediction, metric, and bootstrap value matched exactly. This deterministic reproduction adds no independent samples. The core run took about 32 seconds after source caching; downloads and performance on another device can differ.

A fresh clone downloads only the selected 46 audio/video archive members—about 1.04 GB compressed—plus the 1.17 MB annotation ZIP. Byte-range reads avoid downloading complete archives. To reuse a completed local run, set `POSESPLIT_SOURCE_CACHE` to its directory; files are copied after SHA-256, CRC, and size checks. No symlinks are created. A missing cache triggers official downloads. Failures stop the analysis, and no item is replaced based on its outcome.

The pipeline uses NumPy, Pillow, fsspec, aiohttp, and FFmpeg. It calls no native language/vision models or paid cloud services. Observed package versions are pinned in `requirements.txt`; the launcher installs missing packages only and records the environment used. The measured core-process peak was about 349 MB. The streaming design fits a 48 GB budget; an exact 48 GB laptop measurement has not been made.

## Files

- `pilots/renewed09/pose_split.py`: byte-identical frozen study implementation.
- `pilots/renewed09/reproduce.py` and `run_task09.command`: tested fresh-run orchestration.
- `pilots/renewed09/reviewed_protocol.json` and `work/task09_protocol.json`: identical reviewed protocols.
- `runs/renewed-09-posesplit-20261031/`: observed results, predictions, selected labels, saved features, 15 trained heads, procedural-generation metadata, and all 2,000 paired room-bootstrap draws.
- `work/task09_design_sources/`: official annotation ZIP, original STARSS23 readme/license, and source record metadata.
- `audits/`: pre-execution reviews, full raw-data audit, figure review, and launcher reproduction check.
- `figures/`: two aggregate scientific plots, without participant photographs.
- `PUBLICATION_PROVENANCE.json`: source/public file digests and explicit transformations.

The paper PDF and `overleaf.zip` accompany the release. Paper source files are contained only in the ZIP.

## Scope

Each window has one annotated active event; unannotated interference remains possible. Camera-only reframing leaves microphone audio unchanged. Co-rig virtual rotation re-expresses an existing sound field while rotating the camera view; it is not a new physical microphone recording. The 90° × 90° label describes geometric viewport membership, without certifying occlusion-free visibility or audibility. Both training and test use four fixed headings; unseen-angle and unknown-extrinsic generalization are untested.

Actual video was used only to audit the media/coordinate interface. No image features enter the learned heads, and photo derivatives are omitted here. Views, windows, and seeds remain nested within recordings and rooms. Learned audiovisual fusion, source binding, targeted sampling, and richer controllable generation remain future work.

## Citation and licensing

See `CITATION.cff` and `PREPRINTS_METADATA.txt`. Recommended Preprints subject: **Computer Science and Mathematics**; topic: **Artificial Intelligence and Machine Learning**. This is an author recommendation using verified public browse labels.

New project code is MIT-licensed. STARSS23 material retains its separate MIT notice in `ATTRIBUTION.txt` and `work/task09_design_sources/LICENSE`. Raw recordings, synthetic WAVs, participant-photo derivatives, and foundation-model weights are omitted. The 15 small trained NumPy checkpoints are included.

Artifact: https://github.com/Saurav-K-Aryal/posesplit-pilot

## Manuscript

[Compiled PDF](paper.pdf) · [Overleaf ZIP](overleaf.zip). The ZIP contains draft.tex, bibliography files and figures.
