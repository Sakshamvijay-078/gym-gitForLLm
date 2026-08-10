import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DiskBlobStore } from "../src/diskBlobStore.ts";
import {
  ManifestStore,
  ManifestNotFoundError,
  InvalidParentError,
  MissingShardError,
  AmbiguousHashError,
} from "../src/manifestStore.ts";

function assert(condition: boolean, message: string) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  } else {
    console.log(`ok - ${message}`);
  }
}

async function main() {
  const root = await mkdtemp(join(tmpdir(), "core-versioning-manifest-"));
  const blobStore = new DiskBlobStore(root);
  const manifestStore = new ManifestStore(root, blobStore);

  // --- Root checkpoint (parents: []) ---
  const rootBytes = Buffer.from("root checkpoint weights");
  const rootShardHash = await blobStore.put(rootBytes);

  const rootManifestHash = await manifestStore.commit({
    parents: [],
    shards: [{ hash: rootShardHash, size: rootBytes.length }],
    metadata: { nodeId: "seed", round: 0, timestamp: "2026-08-08T00:00:00Z" },
    validationStatus: "valid",
  });
  assert(typeof rootManifestHash === "string" && rootManifestHash.length === 64, "root manifest commits with parents: []");

  // --- Round 1, single node ---
  const round1Bytes = Buffer.from("round 1 adapter weights");
  const round1ShardHash = await blobStore.put(round1Bytes);

  const round1Hash = await manifestStore.commit({
    parents: [rootManifestHash],
    shards: [{ hash: round1ShardHash, size: round1Bytes.length }],
    metadata: { nodeId: "nodeA", round: 1, timestamp: "2026-08-08T01:00:00Z" },
    validationStatus: "valid",
  });

  const fetchedRound1 = await manifestStore.get(round1Hash);
  assert(fetchedRound1.parents[0] === rootManifestHash, "round 1 manifest correctly points at root as parent");

  // --- Round 2, two nodes train concurrently from round 1 -> branch instead of overwrite ---
  const branchABytes = Buffer.from("round 2 branch A, node B");
  const branchAShardHash = await blobStore.put(branchABytes);
  const branchAHash = await manifestStore.commit({
    parents: [round1Hash],
    shards: [{ hash: branchAShardHash, size: branchABytes.length }],
    metadata: { nodeId: "nodeB", round: 2, timestamp: "2026-08-08T02:00:00Z" },
    validationStatus: "valid",
  });

  const branchBBytes = Buffer.from("round 2 branch B, node C");
  const branchBShardHash = await blobStore.put(branchBBytes);
  const branchBHash = await manifestStore.commit({
    parents: [round1Hash],
    shards: [{ hash: branchBShardHash, size: branchBBytes.length }],
    metadata: { nodeId: "nodeC", round: 2, timestamp: "2026-08-08T02:05:00Z" },
    validationStatus: "valid",
  });

  assert(branchAHash !== branchBHash, "two concurrent commits from the same parent get distinct hashes");

  const children = await manifestStore.children(round1Hash);
  assert(children.length === 2, "round 1 has exactly two children — a real branch, nothing overwritten");

  // --- Lineage walk (first-parent) ---
  const chain = await manifestStore.log(branchAHash);
  assert(chain.length === 3, "log() from branch A walks back 3 manifests: branchA -> round1 -> root");
  assert(chain[0].hash === branchAHash && chain[1].hash === round1Hash && chain[2].hash === rootManifestHash, "log() returns the chain newest-first in the correct order");
  assert(chain[2].parents.length === 0, "the root manifest terminates the chain with parents: []");

  // --- Merge-base detection ---
  const base = await manifestStore.commonAncestor(branchAHash, branchBHash);
  assert(base === round1Hash, "commonAncestor() correctly finds round 1 as the merge-base of both branches");

  const noCommonBase = await manifestStore.commonAncestor(rootManifestHash, "f".repeat(64)).catch(() => "threw");
  assert(noCommonBase === "threw" || noCommonBase === null, "commonAncestor() handles an unrelated/invalid hash without crashing");

  // --- Merge commit: TWO parents ---
  const mergedBytes = Buffer.from("merged weights from branch A + B");
  const mergedShardHash = await blobStore.put(mergedBytes);
  const mergeHash = await manifestStore.commit({
    parents: [branchAHash, branchBHash],
    shards: [{ hash: mergedShardHash, size: mergedBytes.length }],
    metadata: { nodeId: "merger", round: 3, timestamp: "2026-08-08T03:00:00Z", mergeStrategy: "average" },
    validationStatus: "valid",
  });
  const mergeEntry = await manifestStore.get(mergeHash);
  assert(mergeEntry.parents.length === 2, "merge commit stores both parent hashes");

  const branchAChildren = await manifestStore.children(branchAHash);
  const branchBChildren = await manifestStore.children(branchBHash);
  assert(
    branchAChildren.some((e) => e.hash === mergeHash) && branchBChildren.some((e) => e.hash === mergeHash),
    "children() finds the merge commit from EITHER parent branch",
  );

  // --- Idempotent commit ---
  const branchAHashAgain = await manifestStore.commit({
    parents: [round1Hash],
    shards: [{ hash: branchAShardHash, size: branchABytes.length }],
    metadata: { nodeId: "nodeB", round: 2, timestamp: "2026-08-08T02:00:00Z" },
    validationStatus: "valid",
  });
  assert(branchAHashAgain === branchAHash, "committing identical content twice returns the identical hash");

  // --- Hash prefix resolution (git-style) ---
  const resolved = await manifestStore.resolvePrefix(round1Hash.slice(0, 10));
  assert(resolved === round1Hash, "resolvePrefix() resolves a short unambiguous prefix to the full hash");

  let threwAmbiguous = false;
  try {
    await manifestStore.resolvePrefix(""); // empty prefix matches everything
  } catch (err) {
    threwAmbiguous = err instanceof AmbiguousHashError;
  }
  assert(threwAmbiguous, "resolvePrefix() throws AmbiguousHashError when a prefix matches more than one hash");

  // --- Validation guards ---
  let threwInvalidParent = false;
  try {
    await manifestStore.commit({
      parents: ["0".repeat(64)],
      shards: [{ hash: round1ShardHash, size: round1Bytes.length }],
      metadata: { nodeId: "nodeX", round: 3, timestamp: "2026-08-08T03:00:00Z" },
      validationStatus: "valid",
    });
  } catch (err) {
    threwInvalidParent = err instanceof InvalidParentError;
  }
  assert(threwInvalidParent, "commit() rejects a parent hash that doesn't exist");

  let threwMissingShard = false;
  try {
    await manifestStore.commit({
      parents: [round1Hash],
      shards: [{ hash: "1".repeat(64), size: 999 }],
      metadata: { nodeId: "nodeY", round: 3, timestamp: "2026-08-08T03:05:00Z" },
      validationStatus: "valid",
    });
  } catch (err) {
    threwMissingShard = err instanceof MissingShardError;
  }
  assert(threwMissingShard, "commit() rejects a shard hash that isn't in the blob store");

  let threwNotFound = false;
  try {
    await manifestStore.get("2".repeat(64));
  } catch (err) {
    threwNotFound = err instanceof ManifestNotFoundError;
  }
  assert(threwNotFound, "get() on a missing manifest hash throws ManifestNotFoundError");

  await rm(root, { recursive: true, force: true });

  if (process.exitCode === 1) {
    console.error("\nSome checks failed.");
  } else {
    console.log("\nAll checks passed.");
  }
}

main();
