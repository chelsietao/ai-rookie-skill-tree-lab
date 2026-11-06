# AI Engineering Encyclopedia — product specification

This is a **reference atlas**, not a course or a progression tracker. It expands the existing repository rather than replacing its notebooks and reproducible ML experiments.

## Editorial contract for every full article

1. **Plain-language explanation** — intuitive meaning and why the term exists.
2. **Technical mechanism** — concrete data flow, algorithms, key equations, performance trade-offs and failure modes.
3. **Architecture diagram** — a legible schematic (rendered locally, without external image dependencies).
4. **Short code example** — clearly labeled illustrative or executable code; explain environment dependencies.
5. **One industry case** — company/product, concrete problem, how the technology is used, and a public primary source. Distinguish documented applications from a hypothetical example. Do not invent internal engineering details.
6. **Related concepts** — navigable links to nearby terms and prerequisites.
7. **Official documentation** — first-party links and technical specifications.
8. **Further reading** — additional first-party documentation or research.

Do **not** add homework, exams, quests, XP, badges for progress, learning-state persistence, login, or learning analytics.

## Information architecture

- Broad coverage: Programming, Machine Learning, Deep Learning, LLM & Agents, Data & MLOps, GPU Computing, Distributed Systems, Inference Optimization, Compilers.
- Editorial priority: Inference Optimization → Compiler & Fusion → GPU Kernels → Distributed / Training; the rest remain discoverable.
- Explore via text search, category filters, relation graph, and article view.
- An individual article can link to another article without imposing a prerequisite gate.
- Primary sources are curated, and future AI-assisted explanations should cite retrieved sources and never silently overwrite editorial content.

## Implementation

`atlas/index.html`, `atlas/styles.css`, `atlas/data.js`, `atlas/app.js` provide a zero-build static first version. Open `atlas/index.html` in a browser, or serve with `python3 -m http.server 8000` at the repository root and visit `http://localhost:8000/atlas/`.

The initial seed contains **fully written focus articles** plus **clearly labeled glossary entries**. A glossary entry is not represented as a completed full article. This distinction prevents a broad taxonomy from pretending to be a deep encyclopedia.

## Future API integration

A server-side endpoint can accept the current node ID and a user's question, retrieve relevant vetted documentation, and call an LLM. Keep API credentials server-side; never put them in client JavaScript. Responses should carry source links and a clear label distinguishing AI-generated explanations from editorial content.
