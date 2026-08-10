/**
 * A shard reference points into the BlobStore by hash. The checkpoint field
 * is shard-first (a list, even when length 1 today) so scaling to
 * multi-shard checkpoints later doesn't require a schema rewrite.
 */
export interface ShardRef {
  hash: string;
  size: number;
}

export interface ManifestMetadata {
  nodeId: string;
  round: number;
  datasetRef?: string;
  timestamp: string;
  metric?: number;
  /** Set when this manifest is a merge commit — which strategy produced it. */
  mergeStrategy?: string;
  /**
   * Which codec decodes this manifest's shard bytes into ModelWeights —
   * "safetensors" | "json". Undefined means the shard was committed as an
   * opaque blob (e.g. a real .pt/.onnx/.gguf file with no codec yet) —
   * versioning, lineage, and checkout all still work on it fine; only
   * merge needs this field to be set.
   */
  format?: string;
}

export type ValidationStatus = "pending" | "valid" | "invalid";

/**
 * What the caller provides to commit(). Everything except the hash itself —
 * the hash is derived FROM this content, so it can't be part of it.
 *
 * parents is a list, not a single value, on purpose:
 *   []              — root checkpoint, no ancestor
 *   [x]             — ordinary commit, one parent (the common case)
 *   [x, y, ...]     — merge commit, multiple parents (branches combined)
 * This is the same shape git uses for merge commits, and it's what makes
 * merge a natural extension of the lineage graph instead of a special case.
 */
export interface ManifestInput {
  parents: string[];
  shards: ShardRef[];
  metadata: ManifestMetadata;
  validationStatus: ValidationStatus;
}

/**
 * A committed manifest: the input plus its content hash, which is also its
 * identity and its key in the ManifestStore.
 */
export interface ManifestEntry extends ManifestInput {
  hash: string;
}
