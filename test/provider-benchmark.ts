/**
 * Provider Benchmark - Compare Local vs Voyage embeddings
 *
 * Tests retrieval accuracy across providers using the same queries.
 * Run: VOYAGE_API_KEY=xxx bun test/provider-benchmark.ts
 */

import { execSync } from "child_process";
import { existsSync, rmSync } from "fs";
import { join } from "path";

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

const QMD_DIR = "/Users/adam/projects/qmd";

function runCmd(cmd: string, env: Record<string, string> = {}): string {
  const envStr = Object.entries(env).map(([k, v]) => `${k}="${v}"`).join(" ");
  return execSync(`cd ${QMD_DIR} && ${envStr} ${cmd}`, { 
    encoding: "utf-8", 
    timeout: 120000,
    env: { ...process.env, ...env }
  });
}

function runVsearch(query: string, env: Record<string, string>): SearchResult[] {
  try {
    const output = runCmd(
      `bun src/qmd.ts vsearch "${query.replace(/"/g, '\\"')}" --json -n 5 2>/dev/null`,
      env
    );
    return JSON.parse(output);
  } catch (e) {
    return [];
  }
}

function setupProvider(provider: string, indexPath: string): Record<string, string> {
  // Clean up old index
  if (existsSync(indexPath)) {
    rmSync(indexPath);
  }
  
  const env: Record<string, string> = {
    XDG_CACHE_HOME: "/tmp/qmd-benchmark-" + provider,
  };
  
  if (provider === "voyage") {
    env.QMD_PROVIDER = "voyage";
    env.VOYAGE_API_KEY = process.env.VOYAGE_API_KEY || "";
  } else {
    env.QMD_PROVIDER = "local";
  }
  
  console.log(`\nSetting up ${provider} index at ${env.XDG_CACHE_HOME}...`);
  
  // Create collection
  try {
    runCmd(`bun src/qmd.ts collection add test/eval-docs --name eval-docs 2>&1`, env);
  } catch (e) {
    // Collection might already exist
  }
  
  // Embed
  console.log(`Embedding with ${provider}...`);
  runCmd(`bun src/qmd.ts embed 2>&1`, env);
  
  return env;
}

function evaluateProvider(provider: string, env: Record<string, string>): { hit1: number; hit3: number; total: number; latencyMs: number } {
  let hit1 = 0, hit3 = 0;
  let totalLatency = 0;
  
  console.log(`\n=== Evaluating ${provider.toUpperCase()} ===\n`);
  
  for (const { query, expectedDoc, difficulty } of evalQueries) {
    const start = Date.now();
    const results = runVsearch(query, env);
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
  
  // Setup and evaluate Voyage first (faster)
  const voyageEnv = setupProvider("voyage", "/tmp/qmd-benchmark-voyage");
  const voyageResults = evaluateProvider("voyage", voyageEnv);
  
  // Setup and evaluate Local
  const localEnv = setupProvider("local", "/tmp/qmd-benchmark-local");
  const localResults = evaluateProvider("local", localEnv);
  
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
  
  const winner = voyageResults.hit1 > localResults.hit1 ? "🚀 Voyage" : 
                 localResults.hit1 > voyageResults.hit1 ? "🏠 Local" : "🤝 Tie";
  console.log(`Winner (by Hit@1): ${winner}`);
  
  // Cleanup
  console.log("\nCleaning up temp indexes...");
  rmSync("/tmp/qmd-benchmark-voyage", { recursive: true, force: true });
  rmSync("/tmp/qmd-benchmark-local", { recursive: true, force: true });
}

main().catch(console.error);
