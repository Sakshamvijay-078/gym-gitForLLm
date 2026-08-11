import type { MergeStrategy } from "./types.ts";
import { average } from "./strategies/average.ts";
import { taskArithmetic } from "./strategies/taskArithmetic.ts";
import { ties } from "./strategies/ties.ts";
import { slerp } from "./strategies/slerp.ts";
import { confidenceWeighted } from "./strategies/confidenceWeighted.ts";

/**
 * Plug-and-play registry. Adding a new merge technique means writing one
 * file implementing MergeStrategy and adding one line here — nothing that
 * calls getMergeStrategy() needs to change. Same pattern as the
 * allocate() policy interface: the caller depends on the interface, not
 * on any particular implementation.
 */
const strategies: Record<string, MergeStrategy> = {
  [average.name]: average,
  [taskArithmetic.name]: taskArithmetic,
  [ties.name]: ties,
  [slerp.name]: slerp,
  [confidenceWeighted.name]: confidenceWeighted,
};

export function getMergeStrategy(name: string): MergeStrategy {
  const strategy = strategies[name];
  if (!strategy) {
    throw new Error(`Unknown merge strategy "${name}". Available: ${listMergeStrategies().join(", ")}`);
  }
  return strategy;
}

export function listMergeStrategies(): string[] {
  return Object.keys(strategies);
}
