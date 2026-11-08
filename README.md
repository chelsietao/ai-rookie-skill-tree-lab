# AI Engineering Atlas — Interactive Encyclopedia

A production-quality technical knowledge atlas for AI Engineering, from models to systems. Browse 20+ published concepts, explore career pathways, and discover connections between inference optimization, compilers, GPU computing, and distributed systems.

## Features

- **Interactive Knowledge Graph**: Explore technical domains and relationships between concepts
- **Bilingual Search**: Find concepts in English or Traditional Chinese
- **Comprehensive Concepts**: 20+ published technical articles with all 8 required sections
- **Career Pathways**: 7 technical specializations with Singapore and Taiwan market context
- **Responsive Design**: Clean, minimal UI optimized for desktop and mobile

## Getting Started

### Prerequisites
- Node.js 18+ and npm

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Production Build
```bash
npm run build
npm start
```

## Project Structure

```
src/
├── app/                     # Next.js app router
│   ├── page.tsx            # Homepage
│   ├── concepts/           # Concept listing & detail pages
│   ├── careers/            # Career paths explorer
│   └── globals.css         # Global styles
├── components/             # React components
├── data/                   # Concept & career data
├── lib/                    # Utilities
└── types/                  # TypeScript types
```

## Key Concepts

**Inference Optimization**: KV Cache, Continuous Batching, PagedAttention
**Compiler Engineering**: Operator Fusion, torch.compile, Computation Graphs
**GPU Computing**: GPU Memory Hierarchy, CUDA, Triton
**Deep Learning**: Attention, Transformers
**Distributed Systems**: DDP, All-Reduce
**MLOps**: Model Serving

## Career Paths

7 specialized technical directions:
1. Inference Optimization & AI Compiler (Recommended)
2. GPU / CUDA Performance Engineering
3. Distributed Training & AI Systems
4. AI Platform / MLOps
5. Applied Machine Learning / Computer Vision
6. LLM Applications / Agent Engineering
7. AI Infrastructure Operations & Networking

## Language Policy

- **UI & Content**: Traditional Chinese (Taiwan)
- **Technical Terms**: English (preserved)

Example:
```
標題: KV Cache
說明: 在逐 token 生成時保存歷史 token 的 Key/Value，避免重複計算。
```

## Disclaimer

**Career paths evaluations are editorial.** Market claims require current verification. This is not a salary ranking, job guarantee, or immunity from AI automation.

## License

MIT

🤖 Generated with Claude Code
