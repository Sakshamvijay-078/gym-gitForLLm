import { createHash, randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { BlobStore } from "./blobStore.ts";
import type { ManifestEntry, ManifestInput } from "./manifest.ts";

export class ManifestNotFoundError extends Error {
  constructor(public readonly hash: string) {
    super(`Manifest not found: ${hash}`);
    this.name = "ManifestNotFoundError";
  }
}

export class InvalidParentError extends Error {
  constructor(public readonly parentHash: string) {
    super(`Cannot commit: parent manifest does not exist: ${parentHash}`);
    this.name = "InvalidParentError";
  }
}

export class MissingShardError extends Error {
  constructor(public readonly shardHash: string) {
    super(`Cannot commit: shard blob does not exist in blob store: ${shardHash}`);
    this.name = "MissingShardError";
  }
}

export class AmbiguousHashError extends Error {
  constructor(public readonly prefix: string, public readonly matches: string[]) {
    super(`Hash prefix "${prefix}" matches ${matches.length} manifests — be more specific.`);
    this.name = "AmbiguousHashError";
  }
}

/**
 * ManifestStore — tracks metadata + lineage by hash, not raw files.
 *
 * Deliberately separate from BlobStore: blobs are dumb, immutable bytes;
 * manifests are the graph structure (parent pointers) that turns those
 * blobs into a training history. A ManifestStore depends on a BlobStore
 * (to verify shards actually exist before committing) but never the
 * reverse — the blob layer knows nothing about lineage.
 *
 * Manifests live under:
 *   <root>/manifests/<first 2 hash chars>/<remaining hash chars>.json
 */
export class ManifestStore {
  constructor(
    private readonly root: string,
    private readonly blobStore: BlobStore,
  ) {}

  private pathFor(hash: string): string {
    const prefix = hash.slice(0, 2);
    const rest = hash.slice(2);
    return join(this.root, "manifests", prefix, `${rest}.json`);
  }

  private stableStringify(value: unknown): string {
    if (Array.isArray(value)) {
      return `[${value.map((v) => this.stableStringify(v)).join(",")}]`;
    }
    if (value !== null && typeof value === "object") {
      const keys = Object.keys(value as Record<string, unknown>).sort();
      const entries = keys.map(
        (k) => `${JSON.stringify(k)}:${this.stableStringify((value as Record<string, unknown>)[k])}`,
      );
      return `{${entries.join(",")}}`;
    }
    return JSON.stringify(value);
  }

  private hashOf(input: ManifestInput): string {
    const canonical = this.stableStringify(input);
    return createHash("sha256").update(canonical).digest("hex");
  }

  /**
   * Commit a manifest. Validates that every parent already exists and that
   * every shard it references is actually present in the blob store — a
   * manifest can never point at a blob or an ancestor that isn't there.
   *
   * Content-addressed like blobs: committing identical input twice returns
   * the identical hash and doesn't write twice.
   */
  async commit(input: ManifestInput): Promise<string> {
    for (const parentHash of input.parents) {
      if (!(await this.exists(parentHash))) {
        throw new InvalidParentError(parentHash);
      }
    }

    for (const shard of input.shards) {
      if (!(await this.blobStore.exists(shard.hash))) {
        throw new MissingShardError(shard.hash);
      }
    }

    const hash = this.hashOf(input);
    const path = this.pathFor(hash);

    if (existsSync(path)) {
      return hash;
    }

    await mkdir(dirname(path), { recursive: true });

    const entry: ManifestEntry = { ...input, hash };
    const tempPath = join(dirname(path), `.tmp-${randomUUID()}`);
    try {
      await writeFile(tempPath, JSON.stringify(entry, null, 2));
      await rename(tempPath, path);
    } catch (err) {
      await unlink(tempPath).catch(() => {});
      throw err;
    }

    return hash;
  }

  async get(hash: string): Promise<ManifestEntry> {
    const path = this.pathFor(hash);
    if (!existsSync(path)) {
      throw new ManifestNotFoundError(hash);
    }
    const raw = await readFile(path, "utf-8");
    return JSON.parse(raw) as ManifestEntry;
  }

  async exists(hash: string): Promise<boolean> {
    return existsSync(this.pathFor(hash));
  }

  private async allHashes(): Promise<string[]> {
    const manifestsDir = join(this.root, "manifests");
    const prefixes = await readdir(manifestsDir).catch(() => []);
    const hashes: string[] = [];
    for (const prefix of prefixes) {
      const files = await readdir(join(manifestsDir, prefix));
      for (const file of files) {
        if (file.startsWith(".tmp-")) continue;
        hashes.push(prefix + file.replace(/\.json$/, ""));
      }
    }
    return hashes;
  }

  /**
   * Resolve a short hash prefix to a full hash, the way `git rev-parse`
   * resolves an abbreviated SHA. Throws if nothing matches or if more than
   * one manifest shares the prefix.
   */
  async resolvePrefix(prefix: string): Promise<string> {
    if (prefix.length === 64 && (await this.exists(prefix))) {
      return prefix;
    }
    const matches = (await this.allHashes()).filter((h) => h.startsWith(prefix));
    if (matches.length === 0) {
      throw new ManifestNotFoundError(prefix);
    }
    if (matches.length > 1) {
      throw new AmbiguousHashError(prefix, matches);
    }
    return matches[0];
  }

  /**
   * Walk the lineage chain from `hash` back to the root, following the
   * FIRST parent at each step (same convention as `git log --first-parent`).
   * For a merge commit this follows the primary branch, not every ancestor —
   * use commonAncestor() when you need the full merge-base relationship
   * between two tips.
   */
  async log(hash: string): Promise<ManifestEntry[]> {
    const chain: ManifestEntry[] = [];
    let current: string | null = hash;
    while (current !== null) {
      const entry: ManifestEntry = await this.get(current);
      chain.push(entry);
      current = entry.parents[0] ?? null;
    }
    return chain;
  }

  /**
   * Find every manifest that lists `hash` as one of its parents — i.e.
   * every branch (or merge) trained directly off this checkpoint. More
   * than one plain-commit result means concurrent training happened here
   * instead of one node overwriting another.
   */
  async children(hash: string): Promise<ManifestEntry[]> {
    const manifestsDir = join(this.root, "manifests");
    const prefixes = await readdir(manifestsDir).catch(() => []);
    const results: ManifestEntry[] = [];

    for (const prefix of prefixes) {
      const files = await readdir(join(manifestsDir, prefix));
      for (const file of files) {
        if (file.startsWith(".tmp-")) continue;
        const raw = await readFile(join(manifestsDir, prefix, file), "utf-8");
        const entry = JSON.parse(raw) as ManifestEntry;
        if (entry.parents.includes(hash)) {
          results.push(entry);
        }
      }
    }

    return results;
  }

  /**
   * The equivalent of `git merge-base`: the nearest shared ancestor of two
   * tips, found by walking each tip's first-parent chain back to root and
   * finding the first hash they have in common. This is what lets a merge
   * command auto-detect the base for task-arithmetic/TIES without the user
   * having to name it by hand.
   */
  async commonAncestor(hashA: string, hashB: string): Promise<string | null> {
    const chainA = await this.log(hashA);
    const ancestorsA = new Set(chainA.map((e) => e.hash));
    const chainB = await this.log(hashB);
    for (const entry of chainB) {
      if (ancestorsA.has(entry.hash)) {
        return entry.hash;
      }
    }
    return null;
  }
}
