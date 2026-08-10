# @gym/cli

Git-like wrapper over `@gym/core-versioning` (`BlobStore` + `ManifestStore`
+ merge strategies). Auto-discovers the repo the way git does — run any
command from anywhere inside the project, no store path to type.

## Commands

```
gym init
gym commit --file <path> --node <id> --round <n> [--parent <hash>] [--dataset <ref>] [--format <name>]
gym log [<hash>]
gym checkout <hash> --out <path>
gym merge <hashA> <hashB> --strategy <name> --node <id> --round <n> --out <path>
                          [--base <hash>] [--lambda <n>] [--trim <fraction>] [--t <fraction>] [--format <name>]
gym status
gym strategies
gym formats
gym help
```

- `commit` defaults `--parent` to the given node's last commit, or global
  HEAD if the node has never committed — pass `--parent <hash>` explicitly
  to fork a new branch off something other than HEAD.
- `commit` tags the manifest with a model **format**, auto-detected from
  the file extension (`.safetensors`, `.json`) or set explicitly via
  `--format`. Any other file still commits fine — it's just an opaque
  blob until it's tagged or converted, and `merge` will say so clearly if
  you try to merge it.
- Any `<hash>` argument accepts a short prefix, resolved the way
  `git rev-parse` resolves an abbreviated SHA — throws if it's ambiguous.
- `merge` needs both branches to have a recognized format. `--base` is
  optional for task-arithmetic/ties: if omitted, it's auto-detected as
  the nearest common ancestor of the two branches (`git merge-base`).
  Output format defaults to the input format if both branches match, or
  can be set explicitly with `--format`.
- `checkout`/`merge` only support single-shard manifests today —
  multi-shard reassembly comes later, when checkpoints actually need
  sharding.

## Supported model formats

| Format | Read | Write | Notes |
|---|---|---|---|
| `.safetensors` | F32, F64, F16 (upcasts to float32) | F32, F64 | The real HuggingFace binary format — this is what actually matters. |
| `.json` | yes | yes | Synthetic test fixtures only, not a real ML format. |
| `.pt`/`.onnx`/`.gguf` | — | — | Versions and checks out fine as an opaque blob; not mergeable until converted to `.safetensors` (see `core-versioning`'s README for why). |

## Merge strategies

| Strategy | Needs base? | Model count | What it does |
|---|---|---|---|
| `average` | no | 2+ | Plain elementwise mean — the null hypothesis. |
| `task-arithmetic` | yes | 1+ | `base + λ·Σ(model − base)`. `λ` defaults to `1/n` (equal to averaging); push it away from `1/n` to weight branches differently. |
| `ties` | yes | 2+ | Trims each task vector to its top-k magnitude entries, elects a sign per parameter by majority magnitude, averages only the agreeing values. Resolves sign conflicts averaging silently erases. |
| `slerp` | no | exactly 2 | Interpolates along the arc between two weight vectors instead of a straight line — better preserves weight norm than linear blending. `--t` in `[0,1]`, default `0.5`. |

All four are just `MergeStrategy` implementations registered in
`core-versioning/src/merge/registry.ts` — adding a new technique means
writing one file and one registry line, nothing else changes.

## Running it today

Same interim setup as `core-versioning` — no build step yet:

```
node --experimental-transform-types src/index.ts init
```

## Example session

```
gym init
gym commit --file root.safetensors --node seed --round 0
gym commit --file r1.safetensors --node nodeA --round 1
gym commit --file r2a.safetensors --node nodeB --round 2 --dataset shard-cats
gym commit --file r2b.safetensors --node nodeC --round 2 --dataset shard-dogs --parent <r1-hash>
gym log <r2a-hash>
gym merge <r2a-hash> <r2b-hash> --strategy ties --node merger --round 3 --out merged.safetensors
gym status
```

`nodeB` continuing from HEAD and `nodeC` explicitly passing
`--parent <r1-hash>` is what produces a real branch instead of one
overwriting the other — same as checking out a branch in git before you
commit, rather than staying on whatever HEAD currently points to.
