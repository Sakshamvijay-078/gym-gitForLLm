/**
 * BlobStore — content-addressable storage for adapters, checkpoints, and shards.
 *
 * The identity of a blob IS its hash. This gives dedup, integrity checking,
 * and lock-free concurrent writes for free — two nodes that produce the same
 * bytes collapse to one blob without any coordination between them.
 *
 * Nothing above this layer should ever touch the filesystem, S3, or any
 * other storage medium directly. Everything talks to a BlobStore.
 */
export interface BlobStore {
  /**
   * Store bytes and return their content hash. If the bytes already exist
   * in the store, this is a no-op that just returns the existing hash —
   * that's where dedup happens.
   */
  put(bytes: Buffer): Promise<string>;

  /**
   * Retrieve bytes by their hash. Throws BlobNotFoundError if the hash
   * isn't in the store — never returns null/undefined silently, since a
   * caller treating "missing blob" as "empty blob" is a real corruption risk.
   */
  get(hash: string): Promise<Buffer>;

  /**
   * Check whether a hash exists in the store, without reading the bytes.
   * Used by the manifest layer to validate lineage before committing.
   */
  exists(hash: string): Promise<boolean>;
}

export class BlobNotFoundError extends Error {
  constructor(public readonly hash: string) {
    super(`Blob not found: ${hash}`);
    this.name = "BlobNotFoundError";
  }
}
