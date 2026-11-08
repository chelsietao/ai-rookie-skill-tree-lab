import type { CareerPath } from "@/types";
import { compilerExtendedConcepts } from "./concepts/compiler-extended";
import { inferenceExtendedConcepts } from "./concepts/inference-extended";

export const careerPaths: CareerPath[] = [
  {
    id: "inference-compiler",
    title: "Inference Optimization & AI Compiler",
    priority: 1,
    focus: "recommended",
    summary:
      "模型推論、Kernel 與編譯器跨層最佳化，兼顧 Singapore AI infrastructure 與 Taiwan NPU/runtime 職位。",
    markets: {
      singapore:
        "LLM serving / platform / ML systems / inference teams",
      taiwan: "NPU compiler / runtime / edge AI / GPU software",
    },
    skills: [
      "C++",
      "Python",
      "Linux",
      "PyTorch",
      "Transformer",
      "GPU Memory Hierarchy",
      "KV Cache",
      "PagedAttention",
      "Continuous Batching",
      "Quantization",
      "torch.compile",
      "Operator Fusion",
      "LLVM",
      "MLIR",
      "Triton",
    ],
    conceptIds: [
      "cpp",
      "python",
      "attention",
      "gpu-memory",
      "kv-cache",
      "paged-attention",
      "continuous-batching",
      "latency-throughput",
      "torch-compile",
      "operator-fusion",
      "computation-graph",
      "triton",
      "flash-attention",
      "speculative-decoding",
      "quantization",
      "prefix-caching",
      "chunked-prefill",
      "grouped-query-attention",
      "prefill-decode-disaggregation",
      "sampling-decoding",
      "vllm",
      "xla",
      "mlir",
      "onnx",
      "onnx-runtime",
      "tensorrt",
      "torch-inductor",
      ...inferenceExtendedConcepts.map((c) => c.id),
      ...compilerExtendedConcepts.map((c) => c.id),
    ],
    roles: [
      "Inference Engineer",
      "AI Runtime Engineer",
      "AI Compiler Engineer",
      "ML Systems Engineer",
    ],
    tradeoffs:
      "高技術門檻、相關初階職缺較集中；需要效能量測與系統實作證據。",
    assessment:
      "跨市場遷移性與技術護城河高，為編輯判斷，非保證薪資排名。",
  },
  {
    id: "gpu-kernel",
    title: "GPU / CUDA Performance Engineering",
    priority: 2,
    summary:
      "以 CUDA、Triton、記憶體階層和 Profiling 打造高效能 GPU kernels。",
    markets: {
      singapore: "GPU software / AI optimization / accelerated computing",
      taiwan: "Semiconductor software / GPU and NPU performance",
    },
    skills: [
      "C++",
      "CUDA",
      "Triton",
      "Warp",
      "SM",
      "Memory Coalescing",
      "Occupancy",
      "Nsight",
      "GEMM",
    ],
    conceptIds: [
      "cpp",
      "cuda",
      "triton",
      "gpu-memory",
      "operator-fusion",
      "latency-throughput",
    ],
    roles: [
      "GPU Kernel Engineer",
      "CUDA Developer",
      "Performance Engineer",
    ],
    tradeoffs:
      "需要深入硬體知識；職位集中在大型計算與晶片廠商。",
    assessment:
      "技術深度高，市場需求穩定。編輯評估，非薪資承諾。",
  },
  {
    id: "distributed-systems",
    title: "Distributed Training & AI Systems",
    priority: 3,
    summary:
      "跨機器協調、梯度同步、分散式最佳化與大規模 AI 系統架構。",
    markets: {
      singapore:
        "Large-scale AI systems / distributed ML platforms",
      taiwan: "AI infrastructure / distributed computing frameworks",
    },
    skills: [
      "Python",
      "C++",
      "Distributed Systems",
      "NCCL",
      "All-Reduce",
      "DDP",
      "FSDP",
      "ZeRO",
      "Tensor Parallelism",
      "Pipeline Parallelism",
    ],
    conceptIds: [
      "ddp",
      "all-reduce",
      "pytorch",
      "model-serving",
      "gpu-memory",
    ],
    roles: [
      "Distributed Systems Engineer",
      "ML Infrastructure Engineer",
      "AI Platform Engineer",
    ],
    tradeoffs:
      "工程複雜度高；除大型公司外職位較少。",
    assessment:
      "技術可遷移性強。編輯評估，非薪資保證。",
  },
  {
    id: "mlops-platform",
    title: "AI Platform / MLOps",
    priority: 4,
    summary:
      "模型部署、版本管理、監控、CI/CD 與可靠性，支撑生産 AI 系統。",
    markets: {
      singapore: "ML platforms / DevOps / model monitoring",
      taiwan: "AI operations / platform engineering",
    },
    skills: [
      "Python",
      "Docker",
      "Kubernetes",
      "CI/CD",
      "Model Serving",
      "Monitoring",
      "SQL",
      "Model Registry",
    ],
    conceptIds: [
      "python",
      "model-serving",
      "ml-evaluation",
      "latency-throughput",
    ],
    roles: [
      "MLOps Engineer",
      "Platform Engineer",
      "DevOps Engineer",
      "Systems Engineer",
    ],
    tradeoffs:
      "技術層次多樣；需求量大，但職位多為中介層級。",
    assessment:
      "可遷移性高，職場穩定。編輯評估。",
  },
  {
    id: "applied-ml",
    title: "Applied Machine Learning / Computer Vision",
    priority: 5,
    summary:
      "模型選擇、特徵工程、訓練與部署，應用於實務問題。",
    markets: {
      singapore: "Computer vision / image processing / data science",
      taiwan: "Vision systems / autonomous vehicles / imaging",
    },
    skills: [
      "Python",
      "PyTorch",
      "OpenCV",
      "Feature Engineering",
      "Data Annotation",
      "Model Evaluation",
      "SQL",
    ],
    conceptIds: [
      "python",
      "pytorch",
      "ml-evaluation",
      "latency-throughput",
    ],
    roles: [
      "ML Engineer",
      "Computer Vision Engineer",
      "Data Scientist",
    ],
    tradeoffs:
      "入門相對容易；但深度發展需要專業領域知識。",
    assessment:
      "職位最多，但競爭也激烈。編輯評估。",
  },
  {
    id: "llm-agents",
    title: "LLM Applications / Agent Engineering",
    priority: 6,
    summary:
      "大語言模型應用、RAG、提示工程與 AI agents，開發高階應用。",
    markets: {
      singapore: "LLM applications / chatbots / agent systems",
      taiwan: "LLM applications / NLP systems",
    },
    skills: [
      "Python",
      "LLM APIs",
      "Prompt Engineering",
      "RAG",
      "Vector Databases",
      "Tool Calling",
      "Evaluation Metrics",
    ],
    conceptIds: [
      "python",
      "rag",
      "attention",
      "model-serving",
    ],
    roles: [
      "LLM Engineer",
      "AI Agent Engineer",
      "ML Engineer",
    ],
    tradeoffs:
      "門檻低，職位增長快；但技術深度與薪資面臨 AI 自動化風險。",
    assessment:
      "編輯判斷，非薪資或穩定性保證。",
  },
  {
    id: "infrastructure-ops",
    title: "AI Infrastructure Operations & Networking",
    priority: 7,
    summary:
      "GPU 資源管理、網路、叢集調度與營運支撑，維持 AI 系統運行。",
    markets: {
      singapore: "Infrastructure / networking / cluster management",
      taiwan: "Data center / infrastructure operations",
    },
    skills: [
      "Linux",
      "Networking",
      "Kubernetes",
      "Monitoring",
      "RDMA",
      "Cluster Scheduling",
    ],
    conceptIds: [
      "cuda",
      "latency-throughput",
      "model-serving",
    ],
    roles: [
      "Infrastructure Engineer",
      "Site Reliability Engineer",
      "Operations Engineer",
    ],
    tradeoffs:
      "入門門檻相對低；但深化需要網路與系統專長。",
    assessment:
      "職位穩定，技術可遷移。編輯評估，非薪資承諾。",
  },
];
