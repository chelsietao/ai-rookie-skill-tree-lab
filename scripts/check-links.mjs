// 用法：node scripts/check-links.mjs
// 逐一檢查 src/data 內所有來源連結是否可開啟（HTTP 2xx/3xx），並列出失敗者。
// 需要連網；部分網站（如 ACM）可能擋機器人，回 403 時請用瀏覽器手動確認。
import { readFileSync, readdirSync } from "node:fs";

const files = ["src/data/concepts.ts", ...readdirSync("src/data/concepts").map((f) => `src/data/concepts/${f}`)];
const urls = new Map();
for (const f of files) {
  const s = readFileSync(f, "utf8");
  for (const m of s.matchAll(/"(https:\/\/[^"]+)"/g)) {
    if (!urls.has(m[1])) urls.set(m[1], f);
  }
}
console.log(`共 ${urls.size} 個連結`);
const bad = [];
let i = 0;
async function check(url) {
  for (const method of ["HEAD", "GET"]) {
    try {
      const r = await fetch(url, { method, redirect: "follow", signal: AbortSignal.timeout(20000),
        headers: { "user-agent": "Mozilla/5.0 link-check" } });
      if (r.ok) return null;
      if (method === "GET") return `HTTP ${r.status}`;
    } catch (e) {
      if (method === "GET") return String(e.cause?.code ?? e.message);
    }
  }
}
const queue = [...urls.keys()];
await Promise.all(Array.from({ length: 8 }, async () => {
  while (queue.length) {
    const u = queue.shift();
    const err = await check(u);
    i++;
    if (err) { bad.push([u, err, urls.get(u)]); console.log(`✗ ${err}  ${u}  (${urls.get(u)})`); }
  }
}));
console.log(bad.length ? `\n${bad.length} 個連結需要檢查` : "\n全部連結可開啟");
process.exit(bad.length ? 1 : 0);
