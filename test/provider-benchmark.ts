/**
 * Provider Benchmark - Compare embedding providers
 *
 * Tests:
 * 1. Local (embeddinggemma 768d)
 * 2. Voyage 4 large (embed + query)
 * 3. Voyage 4 large (embed) + Voyage 4 nano local (query) - asymmetric
 *
 * Run: VOYAGE_API_KEY=xxx bun test/provider-benchmark.ts
 */

import { execSync } from "child_process";
import { existsSync, rmSync, mkdirSync, writeFileSync } from "fs";
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
const EVAL_DOCS = join(QMD_DIR, "test/eval-docs");

function createIsolatedEnv(name: string): { cacheDir: string; configDir: string; env: Record<string, string> } {
  const baseDir = `/tmp/qmd-benchmark-${name}`;
  const cacheDir = join(baseDir, "cache");
  const configDir = join(baseDir, "config");
  
  // Clean and create dirs
  rmSync(baseDir, { recursive: true, force: true });
  mkdirSync(cacheDir, { recursive: true });
  mkdirSync(configDir, { recursive: true });
  
  // Create minimal config with just eval-docs collection
  const configYaml = `collections:
  eval-docs:
    path: "${EVAL_DOCS}"
    mask: "**/*.md"
`;
  writeFileSync(join(configDir, "index.yml"), configYaml);
  
  return {
    cacheDir,
    configDir,
    env: {
      XDG_CACHE_HOME: cacheDir,
      XDG_CONFIG_HOME: configDir,
      HOME: baseDir, // Override HOME so ~/.config points to our dir
    }
  };
}

function runCmd(cmd: string, extraEnv: Record<string, string> = {}): string {
  const env = { ...process.env, ...extraEnv };
  return execSync(cmd, { 
    cwd: QMD_DIR,
    encoding: "utf-8", 
    timeout: 180000,
    env
  });
}

function runVsearch(query: string, extraEnv: Record<string, string>): SearchResult[] {
  try {
    const output = runCmd(
      `bun src/qmd.ts vsearch "${query.replace(/"/g, '\\"')}" --json -n 5 2>/dev/null`,
      extraEnv
    );
    return JSON.parse(output);
  } catch (e) {
    return [];
  }
}

interface BenchmarkResult {
  name: string;
  hit1: number;
  hit3: number;
  total: number;
  avgLatencyMs: number;
  details: string[];
}

function runBenchmark(name: string, setupEnv: Record<string, string>, queryEnv: Record<string, string>): BenchmarkResult {
  const details: string[] = [];
  let hit1 = 0, hit3 = 0;
  let totalLatency = 0;
  
  console.log(`\n=== ${name} ===\n`);
  
  for (const { query, expectedDoc, difficulty } of evalQueries) {
    const start = Date.now();
    const results = runVsearch(query, queryEnv);
    totalLatency += Date.now() - start;
    
    const ranks = results
      .map((r, i) => ({ rank: i + 1, matches: r.file.toLowerCase().includes(expectedDoc) }))
      .filter(r => r.matches);
    
    const firstHit = ranks.length > 0 ? ranks[0]!.rank : -1;
    
    if (firstHit === 1) hit1++;
    if (firstHit >= 1 && firstHit <= 3) hit3++;
    
    const status = firstHit === 1 ? "✓" : firstHit > 0 ? `@${firstHit}` : "✗";
    const detail = `[${difficulty.padEnd(6)}] ${status.padEnd(3)} "${query}"`;
    details.push(detail);
    console.log(detail);
  }
  
  return { 
    name, 
    hit1, 
    hit3, 
    total: evalQueries.length, 
    avgLatencyMs: totalLatency / evalQueries.length,
    details 
  };
}

async function main() {
  console.log("╔════════════════════════════════════════════════════════════════╗");
  console.log("║            QMD Embedding Provider Benchmark                    ║");
  console.log("╚════════════════════════════════════════════════════════════════╝");
  
  if (!process.env.VOYAGE_API_KEY) {
    console.error("\n❌ VOYAGE_API_KEY not set. Run with:");
    console.error("   VOYAGE_API_KEY=pa-xxx bun test/provider-benchmark.ts");
    process.exit(1);
  }
  
  const results: BenchmarkResult[] = [];
  
  // ========================================
  // 1. LOCAL (embeddinggemma)
  // ========================================
  console.log("\n📦 Setting up LOCAL benchmark...");
  const localSetup = createIsolatedEnv("local");
  const localEnv = { ...localSetup.env, QMD_PROVIDER: "local" };
  
  console.log("   Indexing documents...");
  runCmd("bun src/qmd.ts update 2>&1", localEnv);
  console.log("   Embedding with embeddinggemma (768d)...");
  runCmd("bun src/qmd.ts embed 2>&1", localEnv);
  
  results.push(runBenchmark("LOCAL (embeddinggemma 768d)", localEnv, localEnv));
  
  // ========================================
  // 2. VOYAGE 4 LITE (embed + query)
  // ========================================
  console.log("\n📦 Setting up VOYAGE benchmark...");
  const voyageSetup = createIsolatedEnv("voyage");
  const voyageEnv = { 
    ...voyageSetup.env, 
    QMD_PROVIDER: "voyage",
    VOYAGE_API_KEY: process.env.VOYAGE_API_KEY!
  };
  
  console.log("   Indexing documents...");
  runCmd("bun src/qmd.ts update 2>&1", voyageEnv);
  console.log("   Embedding with voyage-4-lite (1024d)...");
  runCmd("bun src/qmd.ts embed 2>&1", voyageEnv);
  
  results.push(runBenchmark("VOYAGE 4 LITE (1024d)", voyageEnv, voyageEnv));
  
  // ========================================
  // RESULTS SUMMARY
  // ========================================
  console.log("\n" + "═".repeat(66));
  console.log("                         RESULTS SUMMARY");
  console.log("═".repeat(66));
  console.log(`
Provider                      Hit@1     Hit@3     Avg Latency
─────────────────────────────────────────────────────────────`);
  
  for (const r of results) {
    const h1 = ((r.hit1/r.total)*100).toFixed(0).padStart(3);
    const h3 = ((r.hit3/r.total)*100).toFixed(0).padStart(3);
    const lat = r.avgLatencyMs.toFixed(0).padStart(5);
    console.log(`${r.name.padEnd(30)} ${h1}%      ${h3}%      ${lat}ms`);
  }
  
  console.log("─".repeat(66));
  
  // Find winner
  const sorted = [...results].sort((a, b) => b.hit1 - a.hit1);
  console.log(`\n🏆 Winner (by Hit@1): ${sorted[0]?.name || "N/A"}`);
  
  // Cleanup
  console.log("\n🧹 Cleaning up...");
  rmSync("/tmp/qmd-benchmark-local", { recursive: true, force: true });
  rmSync("/tmp/qmd-benchmark-voyage", { recursive: true, force: true });
  
  console.log("\n✅ Benchmark complete!");
}

main().catch(console.error);
