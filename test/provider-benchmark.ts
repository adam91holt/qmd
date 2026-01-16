/**
 * Provider Benchmark - Compare Local vs Voyage embeddings
 *
 * Tests retrieval accuracy across providers using the same queries.
 * Run: VOYAGE_API_KEY=xxx bun test/provider-benchmark.ts
 */

import { execSync } from "child_process";
import { mkdirSync, existsSync, rmSync } from "fs";
import { join } from "path";

const BENCHMARK_INDEX = join(process.env.HOME || "/tmp", ".cache/qmd/benchmark.sqlite");

// Test queries with expected documents
const evalQueries = [
  { query: "API versioning", expectedDoc: "api-design", difficulty: "easy" },
  { query: "Series A fundraising", expectedDoc: "fundraising", difficulty: "easy" },
  { query: "CAP theorem", expectedDoc: "distributed-systems", difficulty: "easy" },
  { query: "how to structure REST endpoints", expectedDoc: "api-design", difficulty: "medium" },
  { query: "raising money for startup", expectedDoc: "fundraising", difficulty: "medium" },
  { query: "consistency vs availability tradeoffs", expectedDoc: "distributed-systems", difficulty: "medium" },
  { query: "nouns not verbs", expectedDoc: "api-design", difficulty: "hard" },
  { query: "Sequoia investor pitch", expectedDoc: "fundraising", difficulty: "hard" },
  { query: "Raft algorithm leader election", expectedDoc: "distributed-systems", difficulty: "hard" },
];

interface SearchResult {
  file: string;
  score: number;
}

function runVsearch(query: string, provider: string, indexPath: string): SearchResult[] {
  const env = provider === "voyage" 
    ? `VOYAGE_API_KEY="${process.env.VOYAGE_API_KEY}" QMD_PROVIDER=voyage`
    : "QMD_PROVIDER=local";
  
  try {
    const output = execSync(
      `cd /Users/adam/projects/qmd && ${env} bun src/qmd.ts --index ${indexPath} vsearch "${query.replace(/"/g, '\\"')}" --json -n 5 2>/dev/null`,
      { encoding: "utf-8", timeout: 60000 }
    );
    return JSON.parse(output);
  } catch (e) {
    return [];
  }
}

function setupIndex(provider: string): string {
  const indexName = `benchmark-${provider}`;
  const indexPath = join(process.env.HOME || "/tmp", ".cache/qmd", `${indexName}.sqlite`);
  
  // Clean up old index
  if (existsSync(indexPath)) {
    rmSync(indexPath);
  }
  
  const env = provider === "voyage"
    ? `VOYAGE_API_KEY="${process.env.VOYAGE_API_KEY}" QMD_PROVIDER=voyage`
    : "QMD_PROVIDER=local";
  
  console.log(`\nSetting up ${provider} index...`);
  
  // Create collection and embed
  execSync(
    `cd /Users/adam/projects/qmd && ${env} bun src/qmd.ts --index ${indexName} collection add test/eval-docs --name eval-docs`,
    { encoding: "utf-8", stdio: "inherit" }
  );
  
  execSync(
    `cd /Users/adam/projects/qmd && ${env} bun src/qmd.ts --index ${indexName} embed`,
    { encoding: "utf-8", stdio: "inherit" }
  );
  
  return indexName;
}

function evaluateProvider(provider: string, indexName: string): { hit1: number; hit3: number; total: number; latencyMs: number } {
  let hit1 = 0, hit3 = 0;
  let totalLatency = 0;
  
  console.log(`\n=== Evaluating ${provider.toUpperCase()} ===\n`);
  
  for (const { query, expectedDoc, difficulty } of evalQueries) {
    const start = Date.now();
    const results = runVsearch(query, provider, indexName);
    totalLatency += Date.now() - start;
    
    const ranks = results
      .map((r, i) => ({ rank: i + 1, matches: r.file.toLowerCase().includes(expectedDoc) }))
      .filter(r => r.matches);
    
    const firstHit = ranks.length > 0 ? ranks[0]!.rank : -1;
    
    if (firstHit === 1) hit1++;
    if (firstHit >= 1 && firstHit <= 3) hit3++;
    
    const status = firstHit === 1 ? "✓" : firstHit > 0 ? `@${firstHit}` : "✗";
    console.log(`[${difficulty.padEnd(6)}] ${status.padEnd(3)} "${query}"`);
  }
  
  return { hit1, hit3, total: evalQueries.length, latencyMs: totalLatency };
}

// Main
async function main() {
  console.log("╔════════════════════════════════════════════════════════╗");
  console.log("║       QMD Provider Benchmark: Local vs Voyage          ║");
  console.log("╚════════════════════════════════════════════════════════╝");
  
  if (!process.env.VOYAGE_API_KEY) {
    console.error("\n❌ VOYAGE_API_KEY not set. Run with:");
    console.error("   VOYAGE_API_KEY=pa-xxx bun test/provider-benchmark.ts");
    process.exit(1);
  }
  
  // Setup indexes
  const localIndex = setupIndex("local");
  const voyageIndex = setupIndex("voyage");
  
  // Evaluate both
  const localResults = evaluateProvider("local", localIndex);
  const voyageResults = evaluateProvider("voyage", voyageIndex);
  
  // Summary
  console.log("\n" + "=".repeat(60));
  console.log("RESULTS SUMMARY");
  console.log("=".repeat(60));
  console.log(`
Provider      Hit@1     Hit@3     Avg Latency
──────────────────────────────────────────────
Local         ${((localResults.hit1/localResults.total)*100).toFixed(0).padStart(3)}%      ${((localResults.hit3/localResults.total)*100).toFixed(0).padStart(3)}%      ${(localResults.latencyMs/localResults.total).toFixed(0)}ms
Voyage        ${((voyageResults.hit1/voyageResults.total)*100).toFixed(0).padStart(3)}%      ${((voyageResults.hit3/voyageResults.total)*100).toFixed(0).padStart(3)}%      ${(voyageResults.latencyMs/voyageResults.total).toFixed(0)}ms
`);
  
  const winner = voyageResults.hit1 > localResults.hit1 ? "Voyage" : 
                 localResults.hit1 > voyageResults.hit1 ? "Local" : "Tie";
  console.log(`Winner (by Hit@1): ${winner}`);
}

main().catch(console.error);
