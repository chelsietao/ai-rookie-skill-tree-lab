# 🤖 AI Engineering Atlas — Interactive Encyclopedia

> A production-quality interactive knowledge encyclopedia for AI Engineering. Explore 100+ concepts across 9 technical domains, discover connections between inference optimization, compilers, GPU computing, distributed systems, and more.

**Live**: [chelsietao.github.io/ai-rookie-skill-tree-lab/](https://chelsietao.github.io/ai-rookie-skill-tree-lab/)

## ✨ Features

- **🎯 100+ Comprehensive Concepts** — Each with 8 chapters:
  - What Is It (基本概念)
  - Why Does It Exist (存在意義)
  - How Does It Actually Work (實現原理)
  - Architecture Diagram (架構圖解)
  - Code Example (程式範例)
  - Real Industry Case (產業案例)
  - Related Concepts (相關概念)
  - Official Resources (官方資源)

- **🌐 Interactive Knowledge Graph** — Visualize relationships between concepts with multiple edge types (prerequisite, related_to, used_in, implements, optimized_by, alternative_to, part_of)

- **🔍 Bilingual Search** — Find concepts in English or Traditional Chinese with real-time filtering

- **📚 9 Technical Domains** — Organized by specialty:
  - Inference Optimization
  - Compiler Engineering
  - GPU Computing
  - Deep Learning
  - Distributed Systems
  - LLM & Agents
  - Training & MLOps
  - Programming & CS
  - Math & ML

- **🎓 7 Career Pathways** — Specialized directions with Singapore/Taiwan market analysis:
  1. Inference Optimization & AI Compiler ⭐ (Recommended)
  2. GPU / CUDA Performance Engineering
  3. Distributed Training & AI Systems
  4. AI Platform / MLOps
  5. Applied Machine Learning / Computer Vision
  6. LLM Applications / Agent Engineering
  7. AI Infrastructure Operations & Networking

- **🎨 Minimalist Design** — Clean white aesthetic, responsive across desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm

### Development
```bash
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000)

### Production
```bash
npm run build
npm start
```

## 📁 Project Structure

```
src/
├── app/                           # Next.js app router
│   ├── page.tsx                   # Homepage with knowledge graph
│   ├── concepts/page.tsx          # All concepts listing
│   ├── concepts/[slug]/page.tsx   # Individual concept details
│   ├── careers/page.tsx           # Career paths explorer
│   └── globals.css                # Global styles
├── components/
│   ├── ConceptCard.tsx            # Concept card component
│   ├── ConceptGraph.tsx           # Knowledge graph visualization
│   ├── ConceptDetail.tsx          # Full concept view
│   └── ...                        # Other UI components
├── data/
│   ├── concepts.ts                # Aggregated concepts
│   ├── concepts/                  # Domain-specific concept files
│   │   ├── inference.ts
│   │   ├── compiler.ts
│   │   ├── gpu.ts
│   │   ├── deep-learning.ts
│   │   ├── distributed.ts
│   │   ├── training.ts
│   │   ├── llm-agents.ts
│   │   ├── mlops.ts
│   │   └── ml-programming.ts
│   └── careerPaths.ts             # 7 career path definitions
├── types/
│   └── index.ts                   # TypeScript interfaces
└── lib/                           # Utilities
```

## 📖 Content Structure

Each concept follows a consistent 8-chapter framework:

```typescript
{
  id: "kv-cache",
  title: "KV Cache",
  domain: "inference-optimization",
  level: "intermediate",
  chapters: {
    what: "KV Cache 是...",
    why: "存在的原因是...",
    how: "實現方式...",
    architecture: "架構圖...",
    codeExample: "Python 程式碼...",
    industryCase: "實際產業應用...",
    relatedConcepts: ["Continuous Batching", "Attention"],
    resources: ["Link 1", "Link 2"]
  },
  status: "published"
}
```

## 🌍 Language Policy

- **UI & Navigation**: Traditional Chinese (Taiwan) — 繁體中文
- **Technical Terms**: English (preserved for accuracy)
- **Content**: Bilingual where applicable

Example:
```
標題: KV Cache
說明: 在逐 token 生成時保存歷史 token 的 Key/Value，避免重複計算。
```

## 🛠 Tech Stack

- **Framework**: Next.js 14+ with TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: React with custom components
- **Data**: TypeScript objects (easily extensible to CMS)
- **Deployment**: GitHub Pages

## 📊 Statistics

- **Concepts**: 100+ across 9 domains
- **Career Paths**: 7 specialized directions
- **Graph Edges**: 200+ relationships mapped
- **Languages**: English + Traditional Chinese
- **Content Status Levels**: Glossary → Draft → Reviewed → Published

## ⚖️ Disclaimer

**Career paths evaluations are editorial.** Market claims require current verification. This is not a salary ranking, job guarantee, or immunity from AI automation.

## 📝 License

MIT

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
