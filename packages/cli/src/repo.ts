import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

export class NotAGymRepoError extends Error {
  constructor() {
    super("not a gym repository (or any parent directory): .gym\nRun `gym init` first.");
    this.name = "NotAGymRepoError";
  }
}

/**
 * Walk up from the current directory looking for a .gym folder, the same
 * way git walks up looking for .git. Means every command after `init` just
 * works from anywhere inside the project — no store path to type or get
 * wrong.
 */
export function findRepoRoot(startDir: string = process.cwd()): string {
  let dir = resolve(startDir);
  while (true) {
    const candidate = join(dir, ".gym");
    if (existsSync(candidate)) {
      return candidate;
    }
    const parent = dirname(dir);
    if (parent === dir) {
      throw new NotAGymRepoError();
    }
    dir = parent;
  }
}
