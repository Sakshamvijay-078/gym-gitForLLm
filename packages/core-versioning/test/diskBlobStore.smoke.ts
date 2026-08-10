import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DiskBlobStore } from "../src/diskBlobStore.ts";
import { BlobNotFoundError } from "../src/blobStore.ts";

function assert(condition: boolean, message: string) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  } else {
    console.log(`ok - ${message}`);
  }
}

async function main() {
  const root = await mkdtemp(join(tmpdir(), "core-versioning-"));
  const store = new DiskBlobStore(root);

  // 1. Round trip: put -> hash -> get returns the same bytes.
  const bytes = Buffer.from("fake adapter weights, round 1");
  const hash = await store.put(bytes);
  assert(typeof hash === "string" && hash.length === 64, "put() returns a 64-char sha256 hash");

  const fetched = await store.get(hash);
  assert(fetched.equals(bytes), "get(hash) returns exactly the bytes that were put");

  const doesExist = await store.exists(hash);
  assert(doesExist === true, "exists(hash) is true after put");

  // 2. Missing hash throws, doesn't return null.
  let threw = false;
  try {
    await store.get("0".repeat(64));
  } catch (err) {
    threw = err instanceof BlobNotFoundError;
  }
  assert(threw, "get() on a missing hash throws BlobNotFoundError");

  const missingExists = await store.exists("0".repeat(64));
  assert(missingExists === false, "exists() is false for a hash that was never stored");

  // 3. Dedup: putting identical bytes twice returns the identical hash and
  // does not create a second object on disk.
  const objectsBefore = await countObjects(root);
  const hash2 = await store.put(Buffer.from("fake adapter weights, round 1"));
  const objectsAfter = await countObjects(root);
  assert(hash2 === hash, "putting identical bytes twice returns the identical hash");
  assert(objectsAfter === objectsBefore, "dedup: no new object written for identical bytes");

  // 4. Different bytes produce a different hash and a new object.
  const hash3 = await store.put(Buffer.from("fake adapter weights, round 2"));
  assert(hash3 !== hash, "different bytes produce a different hash");

  // 5. Directory sharding: the blob actually lives under objects/<2 chars>/<rest>.
  const shardDir = join(root, "objects", hash.slice(0, 2));
  const shardFiles = await readdir(shardDir);
  assert(shardFiles.includes(hash.slice(2)), "blob is stored under git-style sharded directory");

  await rm(root, { recursive: true, force: true });

  if (process.exitCode === 1) {
    console.error("\nSome checks failed.");
  } else {
    console.log("\nAll checks passed.");
  }
}

async function countObjects(root: string): Promise<number> {
  let count = 0;
  const objectsDir = join(root, "objects");
  const prefixes = await readdir(objectsDir).catch(() => []);
  for (const prefix of prefixes) {
    const files = await readdir(join(objectsDir, prefix));
    count += files.length;
  }
  return count;
}

main();
