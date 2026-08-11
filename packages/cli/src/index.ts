#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  DiskBlobStore,
  ManifestStore,
  getMergeStrategy,
  listMergeStrategies,
  getCodecByFormat,
  getCodecByExtension,
  listFormats,
  type ValidationStatus,
  type WeightedModel,
} from "../../core-versioning/src/index.ts";
import { loadRefs, recordCommit } from "./refs.ts";
import { findRepoRoot, NotAGymRepoError } from "./repo.ts";
import { color } from "./color.ts";

function short(hash: string): string {
  return hash.slice(0, 10);
}

function parseFlags(args: string[]): Record<string, string> {
  const flags: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      flags[args[i].slice(2)] = args[i + 1];
      i++;
    }
  }
  return flags;
}

function positionalArgs(args: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--")) {
      i++; // skip its value
      continue;
    }
    out.push(args[i]);
  }
  return out;
}

function fail(message: string): never {
  console.error(color.red(`error: ${message}`));
  process.exit(1);
}

const USAGE = `
${color.bold("gym")} — git for LLMs

${color.bold("Usage")}
  gym init                                       start a new store in this directory
  gym commit --file <path> --node <id> --round <n> [--parent <hash>] [--format <name>] [--dataset-size <n>] [--metric <n>]
                                                  commit a checkpoint/adapter
  gym log [<hash>]                               show lineage, defaults to HEAD
  gym checkout <hash> --out <path>                pull a checkpoint's bytes back out
  gym merge <hashA> <hashB> --strategy <name> --node <id> --round <n> --out <path>
                                                  combine two branches into one
                                                   [--base <hash>|auto] [--lambda <n>] [--trim <fraction>] [--t <fraction>] [--format <name>]
                                                   [--size-weight <n>] [--metric-weight <n>] [--type-trust '{"type":n}']  (confidence-weighted)
                                                   [--ties false]  opt into pure task-vector average (no TIES sign-election)
                                                   [--confidence-temp <n>]  softmax temperature for confidence normalization (default 1.0)
                                                   [--score-mode proportional|sqrt|metric|equal|delta-norm]  how size → confidence (default: sqrt)
  gym status                                      show current store + refs
  gym strategies                                  list available merge strategies
  gym formats                                     list available model codecs

${color.dim("Hashes can be given as a short prefix (like git) — no need to type the full 64 chars.")}
${color.dim("--format is auto-detected from the file extension (.safetensors, .json) when not given.")}
`;

async function cmdInit() {
  const gymDir = join(process.cwd(), ".gym");
  if (existsSync(gymDir)) {
    console.log(color.yellow("Already a gym repository here."));
    return;
  }
  await mkdir(join(gymDir, "objects"), { recursive: true });
  await mkdir(join(gymDir, "manifests"), { recursive: true });
  await writeFile(join(gymDir, "refs.json"), JSON.stringify({ head: null, nodes: {} }, null, 2));
  console.log(color.green(`Initialized empty gym repository in ${gymDir}`));
}

async function cmdCommit(flags: Record<string, string>) {
  const root = findRepoRoot();
  const filePath = flags.file;
  const nodeId = flags.node;
  const round = flags.round;

  if (!filePath || !nodeId || !round) {
    fail("commit requires --file <path> --node <id> --round <n>");
  }
  if (!existsSync(filePath)) {
    fail(`no such file: ${filePath}`);
  }

  const blobStore = new DiskBlobStore(root);
  const manifestStore = new ManifestStore(root, blobStore);
  const refs = await loadRefs(root);

  const bytes = await readFile(filePath);
  const shardHash = await blobStore.put(bytes);

  let parentInput = flags.parent ?? refs.nodes[nodeId] ?? refs.head ?? null;
  if (parentInput) parentInput = await manifestStore.resolvePrefix(parentInput);
  const validationStatus = (flags.status as ValidationStatus) ?? "valid";

  // Best-effort format tagging: recognized by --format, else by extension,
  // else left undefined (an opaque blob — still versions and checks out
  // fine, just isn't decodable by `merge` until it's tagged or converted).
  const format = flags.format ?? getCodecByExtension(filePath)?.format;
  if (flags.format) {
    getCodecByFormat(flags.format); // throws early if the name is bogus
  }

  const manifestHash = await manifestStore.commit({
    parents: parentInput ? [parentInput] : [],
    shards: [{ hash: shardHash, size: bytes.length }],
    metadata: {
      nodeId,
      round: Number(round),
      timestamp: new Date().toISOString(),
      datasetRef: flags.dataset,
      datasetSize: flags["dataset-size"] !== undefined ? Number(flags["dataset-size"]) : undefined,
      metric: flags.metric !== undefined ? Number(flags.metric) : undefined,
      format,
    },
    validationStatus,
  });

  await recordCommit(root, nodeId, manifestHash);

  console.log(
    color.green(`[${nodeId} round ${round}] `) +
      `${short(manifestHash)}` +
      color.dim(`  (parent ${parentInput ? short(parentInput) : "none — root"}, format ${format ?? "unrecognized — not mergeable until tagged"})`),
  );
}

async function cmdLog(positional: string[]) {
  const root = findRepoRoot();
  const blobStore = new DiskBlobStore(root);
  const manifestStore = new ManifestStore(root, blobStore);
  const refs = await loadRefs(root);

  const startInput = positional[0] ?? refs.head;
  if (!startInput) {
    fail("no commits yet, and no hash given.");
  }
  const startHash = await manifestStore.resolvePrefix(startInput);

  const chain = await manifestStore.log(startHash);
  for (const entry of chain) {
    const isMerge = entry.parents.length > 1;
    const label = isMerge ? color.yellow(`merge (${entry.metadata.mergeStrategy})`) : `round ${entry.metadata.round}`;
    console.log(
      `${color.bold(short(entry.hash))}  ${label}  node ${entry.metadata.nodeId}  ${color.dim(entry.metadata.timestamp)}`,
    );
    if (isMerge) {
      console.log(color.dim(`  parents: ${entry.parents.map(short).join(", ")}`));
    }
  }
}

async function cmdCheckout(positional: string[], flags: Record<string, string>) {
  const root = findRepoRoot();
  const hashInput = positional[0];
  const outPath = flags.out;

  if (!hashInput || !outPath) {
    fail("checkout requires <hash> --out <path>");
  }

  const blobStore = new DiskBlobStore(root);
  const manifestStore = new ManifestStore(root, blobStore);
  const hash = await manifestStore.resolvePrefix(hashInput);
  const entry = await manifestStore.get(hash);

  if (entry.shards.length !== 1) {
    fail(`manifest has ${entry.shards.length} shards — multi-shard checkout isn't supported yet.`);
  }

  const bytes = await blobStore.get(entry.shards[0].hash);
  await writeFile(outPath, bytes);
  console.log(color.green(`Checked out ${short(hash)} -> ${outPath}`) + color.dim(` (${bytes.length} bytes)`));
}

async function loadWeightedModel(
  manifestStore: ManifestStore,
  blobStore: DiskBlobStore,
  hashInput: string,
): Promise<{ hash: string; model: WeightedModel; format: string }> {
  const hash = await manifestStore.resolvePrefix(hashInput);
  const entry = await manifestStore.get(hash);
  if (entry.shards.length !== 1) {
    fail(`manifest ${short(hash)} has ${entry.shards.length} shards — merge only supports single-shard checkpoints today.`);
  }
  if (!entry.metadata.format) {
    fail(
      `manifest ${short(hash)} was committed without a recognized model format, so it can't be decoded for merging.\n` +
        `  Available codecs: ${listFormats().join(", ")}.\n` +
        `  Re-commit it with --format <name>, or convert the file to .safetensors first.`,
    );
  }
  const codec = getCodecByFormat(entry.metadata.format);
  const bytes = await blobStore.get(entry.shards[0].hash);
  const weights = codec.decode(bytes);
  const info = {
    datasetSize: entry.metadata.datasetSize,
    validationMetric: entry.metadata.metric,
    datasetType: entry.metadata.datasetRef,
  };
  return { hash, model: { hash, weights, info }, format: entry.metadata.format };
}

async function cmdMerge(positional: string[], flags: Record<string, string>) {
  const root = findRepoRoot();
  const [hashAInput, hashBInput] = positional;
  const strategyName = flags.strategy;
  const nodeId = flags.node;
  const round = flags.round;
  const outPath = flags.out;

  if (!hashAInput || !hashBInput || !strategyName || !nodeId || !round || !outPath) {
    fail("merge requires <hashA> <hashB> --strategy <name> --node <id> --round <n> --out <path>");
  }

  const strategy = getMergeStrategy(strategyName);
  const blobStore = new DiskBlobStore(root);
  const manifestStore = new ManifestStore(root, blobStore);

  const { hash: hashA, model: modelA, format: formatA } = await loadWeightedModel(manifestStore, blobStore, hashAInput);
  const { hash: hashB, model: modelB, format: formatB } = await loadWeightedModel(manifestStore, blobStore, hashBInput);

  let base;
  // Load a base if the strategy requires one, or if the user opted in
  // explicitly with --base <hash> or --base auto (for optional-base
  // strategies like confidence-weighted, whose TIES-style mode only
  // activates when a base is present).
  if (strategy.requiresBase || flags.base) {
    let baseHash = flags.base === "auto" ? undefined : flags.base;
    if (!baseHash) {
      const auto = await manifestStore.commonAncestor(hashA, hashB);
      if (!auto) {
        fail(`"${strategyName}" needs a base and no common ancestor was found for these two hashes — pass --base <hash> explicitly.`);
      }
      baseHash = auto;
      console.log(color.dim(`(auto-detected merge base: ${short(baseHash)})`));
    } else {
      baseHash = await manifestStore.resolvePrefix(baseHash);
    }
    const { model: baseModel } = await loadWeightedModel(manifestStore, blobStore, baseHash);
    base = baseModel.weights;
  }

  const options = {
    base,
    lambda: flags.lambda !== undefined ? Number(flags.lambda) : undefined,
    trimFraction: flags.trim !== undefined ? Number(flags.trim) : undefined,
    t: flags.t !== undefined ? Number(flags.t) : undefined,
    sizeWeight: flags["size-weight"] !== undefined ? Number(flags["size-weight"]) : undefined,
    metricWeight: flags["metric-weight"] !== undefined ? Number(flags["metric-weight"]) : undefined,
    typeTrust: flags["type-trust"] !== undefined ? JSON.parse(flags["type-trust"]) : undefined,
    // --ties false → route confidence-weighted to Mode 2 (task-vector average, no sign election).
    ties: flags["ties"] !== undefined ? flags["ties"] !== "false" : undefined,
    confidenceTemp: flags["confidence-temp"] !== undefined ? Number(flags["confidence-temp"]) : undefined,
    // --score-mode controls how dataset size becomes a confidence signal.
    // Default 'sqrt' is safe for imbalanced (100:1 → 10:1 weight ratio).
    // 'proportional' is the old v2 default that caused confidence collapse.
    scoreMode: flags["score-mode"] as "proportional" | "sqrt" | "metric" | "equal" | "delta-norm" | undefined,
  };

  const mergedWeights = strategy.merge([modelA, modelB], options);

  // Output format: explicit --format wins; otherwise if both branches were
  // the same format, keep that; otherwise infer from --out's extension;
  // otherwise fail rather than silently pick one.
  const outputFormat =
    flags.format ?? (formatA === formatB ? formatA : getCodecByExtension(outPath)?.format);
  if (!outputFormat) {
    fail(
      `branches are different formats (${formatA} vs ${formatB}) and --out's extension doesn't resolve one — pass --format <name> explicitly.`,
    );
  }
  const outputCodec = getCodecByFormat(outputFormat);
  const mergedBytes = outputCodec.encode(mergedWeights);

  await writeFile(outPath, mergedBytes);

  const shardHash = await blobStore.put(mergedBytes);
  const mergeHash = await manifestStore.commit({
    parents: [hashA, hashB],
    shards: [{ hash: shardHash, size: mergedBytes.length }],
    metadata: {
      nodeId,
      round: Number(round),
      timestamp: new Date().toISOString(),
      mergeStrategy: strategyName,
      format: outputFormat,
    },
    validationStatus: "valid",
  });
  await recordCommit(root, nodeId, mergeHash);

  console.log(
    color.green(`[merge:${strategyName}] `) +
      `${short(mergeHash)}` +
      color.dim(`  (${short(hashA)} + ${short(hashB)} -> ${outPath}, ${outputFormat})`),
  );
}

async function cmdStatus() {
  const root = findRepoRoot();
  const refs = await loadRefs(root);
  console.log(`${color.bold("Store:")} ${root}`);
  console.log(`${color.bold("HEAD:")}  ${refs.head ? short(refs.head) : color.dim("(no commits yet)")}`);
  const nodeEntries = Object.entries(refs.nodes);
  if (nodeEntries.length === 0) {
    console.log(color.dim("No node refs yet."));
    return;
  }
  console.log(color.bold("Nodes:"));
  for (const [nodeId, hash] of nodeEntries) {
    console.log(`  ${nodeId.padEnd(12)} ${short(hash)}`);
  }
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const flags = parseFlags(rest);
  const positional = positionalArgs(rest);

  if (!command || command === "help" || command === "--help") {
    console.log(USAGE);
    return;
  }

  try {
    switch (command) {
      case "init":
        await cmdInit();
        break;
      case "commit":
        await cmdCommit(flags);
        break;
      case "log":
        await cmdLog(positional);
        break;
      case "checkout":
        await cmdCheckout(positional, flags);
        break;
      case "merge":
        await cmdMerge(positional, flags);
        break;
      case "status":
        await cmdStatus();
        break;
      case "strategies":
        console.log(listMergeStrategies().join("\n"));
        break;
      case "formats":
        console.log(listFormats().join("\n"));
        break;
      default:
        fail(`unknown command: ${command}\n${USAGE}`);
    }
  } catch (err) {
    if (err instanceof NotAGymRepoError) {
      fail(err.message);
    }
    fail(err instanceof Error ? err.message : String(err));
  }
}

main();
