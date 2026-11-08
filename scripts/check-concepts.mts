import { concepts } from "../src/data/concepts";
import { careerPaths } from "../src/data/careerPaths";

const errors: string[] = [];
const ids = new Set<string>();
for (const c of concepts) {
  if (ids.has(c.id)) errors.push(`duplicate id: ${c.id}`);
  ids.add(c.id);
}
for (const c of concepts) {
  for (const r of c.related) {
    if (!ids.has(r)) errors.push(`${c.id}: related id missing -> ${r}`);
    if (r === c.id) errors.push(`${c.id}: related to itself`);
  }
  if (c.status === "published") {
    for (const k of ["intuition", "mechanism", "diagram", "code", "caseTitle", "case"] as const) {
      const v = c[k];
      if (!v || (Array.isArray(v) && v.length === 0)) errors.push(`${c.id}: missing ${k}`);
    }
    if (c.sources.length < 2) errors.push(`${c.id}: fewer than 2 sources`);
    if (c.related.length < 3) errors.push(`${c.id}: fewer than 3 related`);
  }
  for (const [t, u] of c.sources) {
    if (!/^https:\/\//.test(u)) errors.push(`${c.id}: bad url ${u}`);
    if (!t) errors.push(`${c.id}: empty source title`);
  }
}
for (const p of careerPaths) {
  for (const id of p.conceptIds) if (!ids.has(id)) errors.push(`career ${p.id}: concept missing -> ${id}`);
}
const byCat = new Map<string, number>();
for (const c of concepts) byCat.set(c.category, (byCat.get(c.category) ?? 0) + 1);
console.log("total concepts:", concepts.length);
console.log(Object.fromEntries(byCat));
console.log("status:", concepts.reduce((a: Record<string, number>, c) => ((a[c.status] = (a[c.status] ?? 0) + 1), a), {}));
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("OK: all checks passed");
