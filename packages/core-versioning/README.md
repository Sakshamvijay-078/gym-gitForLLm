# core-versioning

"git for LLMs" — content-addressable blob store, manifest lineage graph,
and pluggable model merging, built format-agnostic from the ground up.

## Architecture: why formats only matter in one place

`BlobStore` and `ManifestStore` never look at the bytes they store — they
version anything: `.safetensors`, `.pt`, `.onnx`, `.gguf`, `.json`,
whatever. Only `merge()` needs an actual in-memory tensor to do math on,
so that's the only layer that knows about file formats, through a small
codec interface:

```
bytes on disk  <-- ModelCodec.decode/encode -->  ModelWeights (Tensor per name)
                                                          |
                                                   MergeStrategy.merge()
```

Adding a new format later (ONNX, GGUF) means writing one file implementing
`ModelCodec` and registering it — nothing in `BlobStore`, `ManifestStore`,
or any merge strategy changes.

## What's here

- `src/blobStore.ts` / `src/diskBlobStore.ts` — content-addressed storage.
  `put`/`get`/`exists`, SHA-256 hashing, git-style sharded directories
  (`objects/ab/cdef...`), atomic writes via temp-file + rename.
- `src/manifest.ts` / `src/manifestStore.ts` — the lineage graph.
  `parents: string[]` (not a single parent) so merge commits — multiple
  parents, same as git — are a natural extension, not a special case.
  `commit()` validates every parent and shard exists before writing.
  `log()` walks first-parent lineage; `children()` finds every branch off
  a checkpoint; `commonAncestor()` is the `git merge-base` equivalent, used
  to auto-detect a merge's base. `resolvePrefix()` resolves short hashes
  like `git rev-parse` does with abbreviated SHAs.
- `src/model.ts` — the canonical in-memory tensor representation
  (`Tensor { dtype, shape, data }`, `ModelWeights = Record<name, Tensor>`)
  every codec decodes into and every merge strategy operates on.
- `src/codecs/` — format <-> `ModelWeights` translation.
  - `safetensors.ts` — a real implementation of HuggingFace's binary
    format (8-byte header length + JSON header + raw tensor bytes).
    Reads F32/F64/F16 (F16 upcasts to float32 on read — same convention
    real merge tools use); writes F32/F64. BF16 and int dtypes aren't
    decoded yet — an honest gap, not a silent one.
  - `json.ts` — a lightweight second codec for synthetic test fixtures.
    Not a real ML interchange format; don't use it for anything that
    matters.
  - `registry.ts` — `getCodecByFormat(name)` / `getCodecByExtension(path)`.
- `src/merge/` — pluggable merge strategies, each a `MergeStrategy`
  implementation registered in `registry.ts`:
  - `average.ts` — elementwise mean, the null-hypothesis baseline.
  - `taskArithmetic.ts` — `base + λ·Σ(model − base)`. The `base` is
    normally `commonAncestor()` from the manifest DAG, so it never has
    to be supplied by guesswork.
  - `ties.ts` — trim to top-k magnitude, elect sign by majority
    magnitude, disjoint-merge only agreeing values. Resolves sign
    conflicts averaging silently erases.
  - `slerp.ts` — spherical interpolation between exactly two models;
    preserves weight norm better than a straight line through weight
    space.

## Formats NOT supported yet, and why

- **`.pt`/`.pth`/`.ckpt`** — these are Python `pickle` streams. Decoding
  pickle safely without executing arbitrary code needs a restricted
  unpickler; not worth building from scratch. Convert to `.safetensors`
  first (the same thing real merge tools like mergekit require).
- **`.onnx`** — protobuf-based, well-defined, but real scope to add
  properly. Reasonable next codec once `.safetensors` is proven out.
- **`.gguf`** — more of an export target for quantized inference than
  something you'd train LoRA-relay adapters into; roadmap item.

All of these still version, track lineage, and check out correctly today
— `format` in the manifest just stays `undefined` for them, meaning
"opaque blob, not mergeable until tagged or converted."

## Running it today

No build step yet — Node 22's TypeScript support runs the `.ts` files
directly, and relative imports use `.ts` extensions accordingly (switch
back to `.js` once a real build step exists — `tsconfig.json` is already
set up for that NodeNext convention).

```
npm test
```

Runs four suites: blob store (9 checks), manifest lineage (17 checks),
merge math (17 checks, verified against hand-computable examples), and
codecs (18 checks, including a byte-level safetensors round-trip and a
hand-built F16 fixture).
