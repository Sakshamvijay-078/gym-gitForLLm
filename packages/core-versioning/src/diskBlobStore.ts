import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile, unlink } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import { BlobNotFoundError, type BlobStore } from "./blobStore.ts";

/**
 * Disk-backed BlobStore. Blobs live under:
 *   <root>/objects/<first 2 hash chars>/<remaining hash chars>
 *
 * Mirrors git's own object-store layout so a few hundred thousand blobs
 * don't collapse into one giant unusable directory.
 */
export class DiskBlobStore implements BlobStore {
  constructor(private readonly root: string) {}

  private pathFor(hash: string): string {
    const prefix = hash.slice(0, 2);
    const rest = hash.slice(2);
    return join(this.root, "objects", prefix, rest);
  }

  private hashOf(bytes: Buffer): string {
    return createHash("sha256").update(bytes).digest("hex");
  }

  async put(bytes: Buffer): Promise<string> {
    const hash = this.hashOf(bytes);
    const finalPath = this.pathFor(hash);

    // Dedup: if it's already there, don't write again.
    if (existsSync(finalPath)) {
      return hash;
    }

    await mkdir(dirname(finalPath), { recursive: true });

    // Write to a temp file first, then rename. Rename is atomic on the same
    // filesystem, so a crash mid-write can never leave a half-written blob
    // sitting at the final hash path.
    const tempPath = join(dirname(finalPath), `.tmp-${randomUUID()}`);
    try {
      await writeFile(tempPath, bytes);
      await rename(tempPath, finalPath);
    } catch (err) {
      // Best-effort cleanup of the temp file if something went wrong.
      await unlink(tempPath).catch(() => {});
      throw err;
    }

    return hash;
  }

  async get(hash: string): Promise<Buffer> {
    const path = this.pathFor(hash);
    if (!existsSync(path)) {
      throw new BlobNotFoundError(hash);
    }
    return readFile(path);
  }

  async exists(hash: string): Promise<boolean> {
    return existsSync(this.pathFor(hash));
  }
}
