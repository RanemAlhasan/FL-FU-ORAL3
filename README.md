# FUSED CIFAR-10 Client-Unlearning Reproduction

A faithful port of the FUSED-Code repository
(`https://github.com/Zhong-Zhengyi/FUSED-Code`) for the paper's Table 1
"Cifar10-ResNet18" client-unlearning experiment, built by reading the
actual source files (`main.py`, `algs/fused_unlearning.py`,
`algs/fl_base.py`, `models/Model_base.py`, `models/CNN_Cifar10.py`,
`dataset/data_utils.py`, `dataset/generate_data.py`, `utils.py`) rather
than the paper's prose, since the two disagree in several places (see
"Known discrepancies" below).

**This is a standalone codebase, independent of the oral-cancer FL/FU
framework** — the two projects share no code, by design, since the real
FUSED-Code implementation differs structurally from how that framework was
built (LoRA via `peft` instead of custom sparse masks, hardcoded critical
layers instead of computed CLI, unweighted FedAvg, a 70/30 per-client
train/test split, etc. — see below).

## Why this differs from the paper's prose

The paper's Eq. 11-23 describe Critical Layer Identification via per-layer
Manhattan-distance "Diff" scores and a randomly-sparsified parameter mask.
**The actual published code does neither of these things for CIFAR-10.**
Concretely, reading `models/Model_base.py::Lora.__init__`:

```python
elif args.data_name == 'cifar10':
    target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                     lora_dropout=0.1, bias="none")
self.lora_model = get_peft_model(global_model, config)
```

The "sparse unlearning adapter" is literally a **LoRA adapter** (via
HuggingFace's `peft` library), and the four target layers are a
**hardcoded list**, not the output of any Diff-score computation. The
`self.param_change_dict`/`compute_diff()` machinery exists in the codebase
but is only used for descriptive CSV logging (`param_change_*.csv`), never
to select which layers get adapters. We replicate the actual, hardcoded
behavior faithfully — see `src/models/resnet_lora.py`.

This reproduction makes the same choice throughout: where the paper's math
and the actual code disagree, **we follow the code**, since the code is
what actually produced Table 1's numbers.

## Known discrepancies between the paper's prose and the actual code

| Paper says | Code actually does |
|---|---|
| Sparse random-mask adapter (Eq. 23) | LoRA adapter via `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)` |
| CLI selects layers via Diff scores (Eq. 11-13) | Layers are a hardcoded literal list per dataset |
| FedAvg weighted by client data volume (implied) | `fl_base.py::fedavg()` is an **unweighted** mean (the weighted-averaging code exists but is commented out) |
| Batch size 128 (Sec 5.1 prose) | `local_batch_size`/`test_batch_size` default to 64 in `main.py` |
| — (not mentioned) | Train/test split for client unlearning is **70/30 per client**, not a fixed external test set — CIFAR-10's own 10k test images get pooled into the federated client data entirely |
| — (not mentioned) | The Byzantine "label flipping" attack is a **deterministic cyclic shift** (`label = (label + 1) % num_classes`), not random reassignment |
| Seed reported as part of experimental setup | `main.py --seed 50` exists as a CLI arg, but the `set_random_seed()` call that would apply it is commented out — **the original authors' own results are not bit-exact reproducible, even by themselves** |

## What's faithfully reproduced vs. what's a documented deviation

**Faithfully reproduced** (matches the source code exactly):
- Dirichlet(α=1.0) partitioning algorithm, including the `least_samples=100`
  per-client retry loop (`src/data/dirichlet_partition.py`)
- Proxy-data carving for MIA shadow training, `proxy_frac=0.2`
  (`src/data/proxy_split.py`)
- 70/30 per-client train/test split for the client-unlearning paradigm
- ResNet18 (`torchvision.models.resnet18(pretrained=True)`) with replaced
  FC head (`src/models/resnet_lora.py`)
- LoRA adapter exact hyperparameters and target layers
- SGD(momentum=0.9, weight_decay=5e-4), lr=0.005 for CIFAR-10
- Unweighted FedAvg
- Byzantine label-shift attack mechanics
- Per-client-per-class RA/FA averaging in `test_client_forget`
- ReA (relearn) and MIA (shadow-model + attack classifier) protocols

**Documented, deliberate deviations** (see `configs/cifar10_client_unlearning.yaml`
comments for the reasoning on each):
- **Epoch counts**: we use the README's quick-start values
  (`global_epoch=100, local_epoch=5`) rather than `main.py`'s bare argparse
  defaults (`global_epoch=2, local_epoch=1`), since 2 rounds cannot plausibly
  produce a meaningful CIFAR-10/ResNet18 result. This is our judgment call,
  not a fact retrieved from the source — change it via `--set` if you want
  to test the bare defaults instead.
- **Random seeding**: we DO apply `torch.manual_seed(args.seed)` in this
  reproduction, even though the original's equivalent call is commented
  out. This makes OUR runs reproducible (re-running with the same config
  gives the same result), at the cost of not matching whatever specific
  random draw produced the original Table 1 numbers — which is impossible
  to match anyway, since the original itself doesn't fix that draw.

## What this means for "matching Table 1"

**Exact numeric match to Table 1 is not achievable, by anyone, including
the original authors**, because their own seeding is disabled. What this
reproduction targets instead is: same algorithm, same architecture, same
hyperparameters, same data pipeline, same evaluation protocol — so that the
*qualitative* pattern Table 1 reports (FUSED's RA/FA close to Retrain's,
at a fraction of the parameter/communication cost) should reproduce, even
though the exact decimal values will differ run-to-run, exactly as they
would for the original authors running their own code twice.

## Usage

```bash
pip install -r requirements.txt

# Full reproduction run (matches paper scale: 50 clients, 100 rounds)
python scripts/run_fused_cifar10.py --config configs/cifar10_client_unlearning.yaml

# Quick smoke test (tiny scale, to verify the pipeline runs before committing
# to the full ~100-round run)
python scripts/run_fused_cifar10.py --config configs/cifar10_client_unlearning.yaml \
    --set num_clients=5 --set global_epochs=2 --set fused_iterations=2 \
    --set relearn_rounds=2 --set local_epochs=1 --set run_mia=false \
    --set pretrained=false --set device=cpu --set num_workers=0
```

`dataset_root` in the config should point at your existing
`dataset/cifar10` folder (containing `cifar-10-batches-py/`) — the loader
skips downloading if that folder is already present.

### Overriding any setting

Every value in the YAML config can be overridden via repeatable `--set
key=value` flags, same convention as the oral-cancer framework:

```bash
python scripts/run_fused_cifar10.py --config configs/cifar10_client_unlearning.yaml \
    --set forget_client_idx=[3] --set n_shadow=3 --set alpha=0.1
```

## Output

Each run produces, under `logs/<run_id>/`:
- `config.snapshot.yaml` — fully-resolved config (including `--set` overrides)
- `train.log` — full text log
- `metrics.json` — all scalars + a `"final"` block with the headline
  RA/FA/ReA/MIA numbers for both FUSED and Retrain, namespaced as
  `eval/unlearning/FUSED/{RA,FA,ReA,MIA_acc}` and
  `eval/unlearning/Retrain/{RA,FA,ReA,MIA_acc}`
- `tb/` — TensorBoard event file

And under `checkpoints/<run_id>/`: `phase_a_model.pt`, `fused_model.pt`,
`retrain_model.pt`.

The script's final stdout/log output prints a Table-1-style side-by-side
summary directly.

## Runtime expectations

At full paper scale (50 clients, 100 global rounds, 100 FUSED iterations,
100 relearn rounds, 5 MIA shadow models each re-running the ENTIRE
train_normal + forget_client_train pipeline) — this is substantial compute.
The MIA step alone is `n_shadow` (5) full re-runs of Phase A + Phase B for
each of FUSED and Retrain (10 full extra pipeline runs total). Budget for
a long-running GPU job; consider setting `run_mia: false` for an initial
pass to get RA/FA/ReA quickly, then enabling MIA separately once those
look right.
