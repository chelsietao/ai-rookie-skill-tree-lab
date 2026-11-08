import type { ConceptPaper } from "@/types";

export const existingConceptPapers: Record<string, ConceptPaper[]> = {
  "kv-cache": [
    {
      title: "Attention Is All You Need",
      venue: "NeurIPS",
      year: 2017,
      url: "https://arxiv.org/abs/1706.03762",
      note: "提出 Transformer 與 Q/K/V 注意力結構，KV Cache 快取的正是其中的 Key 與 Value。",
    },
    {
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      venue: "SOSP",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "分析 KV Cache 造成的記憶體碎片與浪費，是討論快取管理最常引用的論文。",
    },
  ],
  "continuous-batching": [
    {
      title: "Orca: A Distributed Serving System for Transformer-Based Generative Models",
      venue: "OSDI",
      year: 2022,
      url: "https://www.usenix.org/conference/osdi22/presentation/yu",
      note: "提出 iteration-level scheduling，是 continuous batching 的原始來源。",
    },
  ],
  "paged-attention": [
    {
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      venue: "SOSP",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "提出 PagedAttention 與 vLLM 的原始論文。",
    },
  ],
  "latency-throughput": [
    {
      title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
      venue: "OSDI",
      year: 2024,
      url: "https://arxiv.org/abs/2403.02310",
      note: "說明 prefill 與 decode 互相干擾時，延遲與吞吐量如何此消彼長。",
    },
  ],
  "flash-attention": [
    {
      title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
      venue: "NeurIPS",
      year: 2022,
      url: "https://arxiv.org/abs/2205.14135",
      note: "FlashAttention 的原始論文，以減少 GPU 記憶體讀寫為核心。",
    },
    {
      title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
      venue: "ICLR",
      year: 2024,
      url: "https://arxiv.org/abs/2307.08691",
      note: "改善平行化與工作切分的第二版。",
    },
  ],
  "speculative-decoding": [
    {
      title: "Fast Inference from Transformers via Speculative Decoding",
      venue: "ICML",
      year: 2023,
      url: "https://arxiv.org/abs/2211.17192",
      note: "提出以小模型草擬、大模型驗證，並維持輸出分布不變。",
    },
    {
      title: "Accelerating Large Language Model Decoding with Speculative Sampling",
      venue: "arXiv",
      year: 2023,
      url: "https://arxiv.org/abs/2302.01318",
      note: "同時期獨立提出的 speculative sampling。",
    },
  ],
  quantization: [
    {
      title: "A White Paper on Neural Network Quantization",
      venue: "arXiv",
      year: 2021,
      url: "https://arxiv.org/abs/2106.08295",
      note: "整理量化的基本原理與實務流程，適合入門。",
    },
  ],
  "prefix-caching": [
    {
      title: "SGLang: Efficient Execution of Structured Language Model Programs",
      venue: "NeurIPS",
      year: 2024,
      url: "https://arxiv.org/abs/2312.07104",
      note: "以 RadixAttention 在多個請求之間重用共同前綴的 KV Cache。",
    },
  ],
  "chunked-prefill": [
    {
      title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
      venue: "OSDI",
      year: 2024,
      url: "https://arxiv.org/abs/2403.02310",
      note: "提出 chunked prefill 與 stall-free scheduling。",
    },
  ],
  "grouped-query-attention": [
    {
      title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
      venue: "EMNLP",
      year: 2023,
      url: "https://arxiv.org/abs/2305.13245",
      note: "GQA 的原始論文，說明如何在品質與 KV Cache 大小之間取捨。",
    },
  ],
  "prefill-decode-disaggregation": [
    {
      title: "Splitwise: Efficient Generative LLM Inference Using Phase Splitting",
      venue: "ISCA",
      year: 2024,
      url: "https://arxiv.org/abs/2311.18677",
      note: "把 prefill 與 decode 分開放在不同機器上。",
    },
    {
      title: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving",
      venue: "OSDI",
      year: 2024,
      url: "https://arxiv.org/abs/2401.09670",
      note: "以符合 SLO 的 goodput 為目標設計分離式架構。",
    },
  ],
  "sampling-decoding": [
    {
      title: "The Curious Case of Neural Text Degeneration",
      venue: "ICLR",
      year: 2020,
      url: "https://arxiv.org/abs/1904.09751",
      note: "提出 nucleus（top-p）sampling。",
    },
  ],
  vllm: [
    {
      title: "Efficient Memory Management for Large Language Model Serving with PagedAttention",
      venue: "SOSP",
      year: 2023,
      url: "https://arxiv.org/abs/2309.06180",
      note: "vLLM 的原始論文。",
    },
  ],
  "operator-fusion": [
    {
      title: "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning",
      venue: "OSDI",
      year: 2018,
      url: "https://arxiv.org/abs/1802.04799",
      note: "在圖層級依規則融合運算子，並以成本模型選擇實作。",
    },
  ],
  "torch-compile": [
    {
      title: "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation",
      venue: "ASPLOS",
      year: 2024,
      url: "https://doi.org/10.1145/3620665.3640366",
      note: "說明 torch.compile 背後的 TorchDynamo 與 TorchInductor。",
    },
  ],
  "torch-inductor": [
    {
      title: "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation",
      venue: "ASPLOS",
      year: 2024,
      url: "https://doi.org/10.1145/3620665.3640366",
      note: "TorchInductor 的設計與 kernel 生成流程。",
    },
  ],
  "computation-graph": [
    {
      title: "TensorFlow: A System for Large-Scale Machine Learning",
      venue: "OSDI",
      year: 2016,
      url: "https://arxiv.org/abs/1605.08695",
      note: "以資料流圖表示運算，是靜態計算圖的代表。",
    },
    {
      title: "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
      venue: "NeurIPS",
      year: 2019,
      url: "https://arxiv.org/abs/1912.01703",
      note: "define-by-run 動態圖的代表。",
    },
  ],
  mlir: [
    {
      title: "MLIR: A Compiler Infrastructure for the End of Moore's Law",
      venue: "arXiv",
      year: 2020,
      url: "https://arxiv.org/abs/2002.11054",
      note: "MLIR 的設計理念與架構。",
    },
  ],
};
