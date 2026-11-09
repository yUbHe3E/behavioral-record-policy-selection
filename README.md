# Relational Behavioral Records for Requirement-Conditioned Language Model Policy Selection

Code and retained evidence for the revised journal manuscript of the same title.
This repository studies the behavioral information supplied to a language-model
policy selector. A deterministic compiler turns validation trajectories into
records; the selector receives those records and a natural-language requirement.
Separate reference procedures evaluate its choices on held-out evidence.

The controlled scenarios provide the principal evidence. The usefulness of
preserved relations varies across requirements and model–prompt configurations.
LunarLander and Highway provide auxiliary tests with trained policies; their
results do not establish a general improvement across environments.

## Quick start: no API key

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
python -m pip install -e ".[dev,plots]"
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test generates a small synthetic example. It does not reproduce the
paper by itself, train policies, or contact a model service.

## Reproduce the reported results

1. Download `evidence-v1.0.0.zip` from the
   [v1.0.0 release](https://github.com/yUbHe3E/behavioral-record-policy-selection/releases/tag/v1.0.0).
2. Extract it into the repository root. This adds `outputs/`, `sha256.json`, and
   `evidence_manifest.json`.
3. Run:

```bash
python scripts/reproduce_all.py
python scripts/plot_revision_results.py
```

The first command verifies the evidence hashes and recomputes the original primary
comparisons, the additional controlled comparisons, compact-record retention, and
both external studies from the retained responses. It also checks their numerical
results against the archived analyses. Results go to `reproduced/` and
`reproduced_primary/`; four result figures go to `generated_figures/`.
The evidence archive is about 300 MB compressed. No credentials or new paid
requests are needed.

See [REPRODUCING.md](REPRODUCING.md) for individual commands, the mapping to the
paper, and the limits of each reproduction route.

## Repository contents

| Location | Contents |
|---|---|
| `src/behavioral_records/` | Record construction, controlled simulator, environment adapters, reference procedures, selector interfaces, and evaluation |
| `configs/` | Experiment and representation configurations |
| `schemas/` | Trajectory and record contracts |
| `scripts/` | Evidence preparation, offline analysis, examples, and selected training/model-call entry points |
| `tests/` | Offline tests for compiler behavior, data separation, response handling, and statistical units |
| `figure_data/` | Numerical inputs used by the result-figure script |
| Release evidence archive | Frozen comparisons, validation/reference trajectories, retained responses, failures, and final analyses |

Anonymous candidate IDs and original condition identifiers are retained to keep
the evidence traceable. `C1` denotes Recovery-Contingent Repetition; `C2` denotes
Necessity-Conditioned Safety and State Preservation. `Relational-Linked` and
`Relational-LinkBroken` denote preserved and disrupted relations. Evaluation-only
mappings and held-out reference files are not inputs to the model judge.

## Re-running training or model calls

Training and model calls are optional and are **not** triggered by the commands
above. LunarLander PPO training is implemented in
`scripts/run_lunarlander_policy_pool.py`; Highway DQN training and qualification
are implemented in `scripts/train_highway_policy.py` and
`src/behavioral_records/bounded_external.py`. Install the corresponding
`lunarlander` or `highway` extra before using these modules.

Trained neural weights are not included in this release. Stored trajectories
support the reported offline evaluations; reproducing training requires a new run
and is not promised to reproduce individual stochastic checkpoints exactly.
Historical configurations and evidence remain fixed. Model services may have
changed since the experiments, so a fresh call is a new evaluation, not a recovery
of the archived response. Credentials must be supplied through environment
variables and must never be committed.

## Data and version information

All released trajectories describe simulated tasks; policy identifiers do not
identify human participants. Host-specific paths and account/transport metadata
have been removed from the public copy. Frozen judge request files retain their
original bytes. Original experiments retained parsed final answers; the later
studies also retain provider responses. Missing historical response bodies have
not been reconstructed. Failure records and negative external results are included.

The release manifest distinguishes original-file hashes from public-file hashes
where metadata redaction was necessary. See [RELEASE_NOTES.md](RELEASE_NOTES.md).
Code is provided under the [MIT license](LICENSE). Cite the manuscript title and
the versioned repository while publication details are pending; no DOI is claimed.
