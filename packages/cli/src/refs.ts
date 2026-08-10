import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

/**
 * refs.json — the CLI's own bookkeeping, separate from the manifest store
 * itself. Tracks a global HEAD (last commit made through this CLI) and a
 * per-node HEAD, so `commit` doesn't require typing out a 64-char parent
 * hash every single time.
 */
export interface Refs {
  head: string | null;
  nodes: Record<string, string>;
}

function refsPath(storeDir: string): string {
  return join(storeDir, "refs.json");
}

export async function loadRefs(storeDir: string): Promise<Refs> {
  const path = refsPath(storeDir);
  if (!existsSync(path)) {
    return { head: null, nodes: {} };
  }
  const raw = await readFile(path, "utf-8");
  return JSON.parse(raw) as Refs;
}

export async function saveRefs(storeDir: string, refs: Refs): Promise<void> {
  await writeFile(refsPath(storeDir), JSON.stringify(refs, null, 2));
}

export async function recordCommit(storeDir: string, nodeId: string, hash: string): Promise<void> {
  const refs = await loadRefs(storeDir);
  refs.head = hash;
  refs.nodes[nodeId] = hash;
  await saveRefs(storeDir, refs);
}
