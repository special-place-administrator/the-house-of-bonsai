import { performWebSearchRaw, performLocalSearchRaw } from "./src/utils/braveApi.js";
import { runInSandbox } from "./src/utils/executor.js";

// Ensure environment variables are loaded
import * as dotenv from "dotenv";
dotenv.config();

async function runBenchmark() {
  console.log("=== Code Mode Benchmark ===");
  const query = "latest machine learning papers 2026";
  const count = 20; // max results
  const offset = 0;

  console.log(`\nScenario: Searching for "${query}" (count: ${count})`);

  // 1. Fetch raw data
  const startFetch = Date.now();
  const rawDataStr = await performWebSearchRaw(query, count, offset);
  const fetchTime = Date.now() - startFetch;

  const beforeSizeKB = Buffer.byteLength(rawDataStr, 'utf8') / 1024;
  console.log(`[Raw Payload] fetch time: ${fetchTime}ms, size: ${beforeSizeKB.toFixed(2)} KB`);

  // 2. Define our extraction code (simulate LLM writing a specific script)
  // E.g., The LLM only wants titles and descriptions of the results.
  const code = `
    try {
      const data = JSON.parse(DATA);
      if (data.web && data.web.results) {
        const extracted = data.web.results.map(r => ({
          title: r.title,
          url: r.url,
          description: r.description
        }));
        console.log(JSON.stringify(extracted, null, 2));
      } else {
        console.log("No web results found.");
      }
    } catch (e) {
      console.log("Error processing data: " + e.message);
    }
  `;

  // 3. Run Code Mode
  const { stdout, error, executionTimeMs } = await runInSandbox(rawDataStr, code);

  if (error) {
    console.error("Sandbox Execution Error:", error);
    return;
  }

  const output = stdout.trim();
  const afterSizeKB = Buffer.byteLength(output, 'utf8') / 1024;

  console.log(`[Code Mode] extract time: ${executionTimeMs}ms, size: ${afterSizeKB.toFixed(2)} KB`);
  console.log(`\nReduction: ${(100 - (afterSizeKB / beforeSizeKB) * 100).toFixed(2)}%`);

  console.log("\n--- Extracted Output Snippet ---");
  console.log(output.substring(0, 500) + (output.length > 500 ? "\n... (truncated)" : ""));
  console.log("--------------------------------\n");

  console.log("=== Local Search Code Mode Benchmark ===\n");
  const docQuery = "pizza near Times Square NY";
  const docCount = 10;

  console.log(`Scenario: Local searching for "${docQuery}" (count: ${docCount})`);

  const startFetchL = Date.now();
  const rawDataStrL = await performLocalSearchRaw(docQuery, docCount);
  const fetchTimeL = Date.now() - startFetchL;

  const beforeSizeKBL = Buffer.byteLength(rawDataStrL, 'utf8') / 1024;
  console.log(`[Raw Payload] fetch time: ${fetchTimeL}ms, size: ${beforeSizeKBL.toFixed(2)} KB`);

  const codeL = `
    try {
      var data = JSON.parse(DATA);
      if (data.source === "local" && data.poisData && data.poisData.results && data.poisData.results.length > 0) {
        var extracted = [];
        for (var i = 0; i < data.poisData.results.length; i++) {
          var r = data.poisData.results[i];
          extracted.push({
            name: r.name,
            phone: r.phone,
            rating: r.rating ? r.rating.ratingValue : null,
            address: r.address ? r.address.streetAddress : null
          });
        }
        console.log(JSON.stringify(extracted, null, 2));
      } else if (data.source === "web_fallback" && data.formattedText) {
        console.log("Web fallback: " + data.formattedText.substring(0, 300));
      } else {
        console.log("Source: " + data.source + ", keys: " + Object.keys(data).join(", "));
      }
    } catch (e) {
      console.log("Error: " + e.message);
    }
    `;

  const { stdout: stdoutL, error: errorL, executionTimeMs: execTimeL } = await runInSandbox(rawDataStrL, codeL);

  if (errorL) {
    console.error("Sandbox Execution Error:", errorL);
    return;
  }

  const outputL = stdoutL.trim();
  const afterSizeKBL = Buffer.byteLength(outputL, 'utf8') / 1024;

  console.log(`[Code Mode] extract time: ${execTimeL}ms, size: ${afterSizeKBL.toFixed(2)} KB`);
  console.log(`\nReduction: ${(100 - (afterSizeKBL / beforeSizeKBL) * 100).toFixed(2)}%`);

  console.log("\n--- Extracted Output Snippet ---");
  console.log(outputL.substring(0, 500) + (outputL.length > 500 ? "\n... (truncated)" : ""));
  console.log("--------------------------------\n");

  // === 3. Generic code_mode_transform benchmark (simulated GitHub issues payload) ===
  console.log("=== Generic code_mode_transform Benchmark ===\n");
  console.log("Scenario: Simulated GitHub list_issues payload (50 issues with full metadata)\n");

  // Generate a realistic mock GitHub issues payload
  const mockIssues = Array.from({ length: 50 }, (_, i) => ({
    id: 1000 + i,
    number: 100 + i,
    title: `Issue ${100 + i}: ${["Fix login bug", "Add dark mode", "Update deps", "Refactor API", "Memory leak in worker"][i % 5]}`,
    state: i % 3 === 0 ? "closed" : "open",
    user: { login: `dev${i % 10}`, id: 5000 + i, avatar_url: `https://avatars.example.com/${i}`, html_url: `https://github.com/dev${i % 10}` },
    labels: [{ id: i, name: ["bug", "enhancement", "docs", "urgent", "help wanted"][i % 5], color: "fc2929" }],
    assignees: [{ login: `dev${(i + 1) % 10}`, id: 6000 + i }],
    body: `Full description of issue ${100 + i}. `.repeat(20) + `\n\nSteps to reproduce:\n1. Step one\n2. Step two\n3. Step three\n\nExpected: X\nActual: Y`,
    created_at: "2026-03-01T10:00:00Z",
    updated_at: "2026-03-08T15:30:00Z",
    closed_at: i % 3 === 0 ? "2026-03-09T09:00:00Z" : null,
    comments: Math.floor(Math.random() * 20),
    reactions: { "+1": Math.floor(Math.random() * 10), "-1": 0, laugh: 0, hooray: 0, confused: 0, heart: 0, rocket: 0, eyes: 0 },
    html_url: `https://github.com/org/repo/issues/${100 + i}`,
    pull_request: null,
    milestone: i % 7 === 0 ? { title: "v2.0", due_on: "2026-04-01" } : null,
  }));
  const mockPayload = JSON.stringify(mockIssues);
  const beforeG = Buffer.byteLength(mockPayload, 'utf8') / 1024;
  console.log(`[Simulated Payload] size: ${beforeG.toFixed(2)} KB`);

  const codeG = `
  var issues = JSON.parse(DATA);
  var summary = [];
  for (var i = 0; i < issues.length; i++) {
    var is = issues[i];
    summary.push("#" + is.number + " [" + is.state + "] " + is.title + " (" + is.comments + " comments)");
  }
  console.log(summary.join("\\n"));
  `;

  const { stdout: stdoutG, error: errorG, executionTimeMs: execTimeG } = await runInSandbox(mockPayload, codeG);

  if (errorG) {
    console.error("Sandbox Execution Error:", errorG);
    return;
  }

  const outputG = stdoutG.trim();
  const afterG = Buffer.byteLength(outputG, 'utf8') / 1024;

  console.log(`[Code Mode] extract time: ${execTimeG}ms, size: ${afterG.toFixed(2)} KB`);
  console.log(`\nReduction: ${(100 - (afterG / beforeG) * 100).toFixed(2)}%`);

  console.log("\n--- Extracted Output Snippet ---");
  console.log(outputG.substring(0, 600) + (outputG.length > 600 ? "\n... (truncated)" : ""));
  console.log("--------------------------------\n");
}

runBenchmark().catch(console.error);
