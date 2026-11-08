import type { Concept } from "@/types";

export const concepts: Concept[] = [
  {
    id: "kv-cache",
    title: "KV Cache",
    zh: "鍵值快取",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "在逐 token 生成時保存歷史 token 的 Key/Value，避免重複計算。",
    intuition:
      "像讀一本書時保留先前做好的索引，不必每次回答都從第一頁重新整理。注意它不是把完整 Attention 結果直接快取。",
    mechanism: [
      "Decoder-only Transformer 每次生成新 token，新的 Query 需要與上下文中已存在的 Keys/Values 計算 Attention。",
      "不使用快取時，每一步會重新計算歷史位置的 K、V。使用快取後只為新增 token 計算 K、V，再與已有 cache 串接或寫入預分配緩衝區。",
      "每層典型 K/V 佔用量約為 2 × batch × sequence_length × num_kv_heads × head_dim × bytes_per_element。GQA/MQA 能減少 KV heads 及快取需求。",
      "代價是推論過程中佔用持續增長的 GPU 記憶體。管理策略、分頁、序列長度限制與 eviction 都會影響服務容量。",
    ],
    diagram: [
      "Input tokens",
      "Q / K / V projection",
      "Store new K/V in cache",
      "Attention: Q with all cached K/V",
      "Next token",
    ],
    code: `// Illustrative PyTorch-style pseudocode
new_k, new_v = project_kv(new_token)
k_cache = torch.cat([k_cache, new_k], dim=-2)
v_cache = torch.cat([v_cache, new_v], dim=-2)
output = attention(query, k_cache, v_cache)
// Production systems avoid repeated torch.cat allocations.`,
    caseTitle: "vLLM — PagedAttention",
    case: "vLLM 公開介紹了 PagedAttention：將 KV Cache 管理成可分配的 blocks，以降低碎片與浪費、提升並行服務容量。這是已公開的系統設計，不代表所有公司使用相同實作。",
    related: ["paged-attention", "continuous-batching", "attention", "gpu-memory"],
    sources: [
      ["vLLM PagedAttention paper", "https://arxiv.org/abs/2309.06180"],
      [
        "Hugging Face KV cache guide",
        "https://huggingface.co/docs/transformers/kv_cache",
      ],
    ],
  },
  {
    id: "continuous-batching",
    title: "Continuous Batching",
    zh: "連續批次排程",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "在推論服務中以迭代為單位動態加入與移除請求，而不是等整批完成。",
    intuition:
      "像餐廳不必等同一桌所有人吃完才接待下一位客人；完成的請求可以離開，新的請求隨時補進。",
    mechanism: [
      "靜態 batching 常因不同生成長度而讓先完成的 request 佔住整個 batch 的同步節點。",
      "Iteration-level scheduling 每個 decode iteration 更新 active requests，讓新請求進入並讓完成的請求離開。",
      "Scheduler 需要考慮 prefill/decode、token budget、KV cache 容量、優先權與公平性。",
      "吞吐量提升並不保證每個 request 的尾端延遲下降；需要量測 TTFT、TPOT、p95/p99 與佇列延遲。",
    ],
    diagram: [
      "Incoming request queue",
      "Token-budget scheduler",
      "Prefill / decode active batch",
      "GPU model execution",
      "Stream generated tokens",
    ],
    code: `# Simplified scheduling illustration
for step in range(max_steps):
    active = remove_finished(active)
    active += admit_requests(queue, token_budget)
    outputs = model_decode_step(active)
    stream_to_clients(outputs)`,
    caseTitle: "vLLM — Continuous Batching",
    case: "vLLM 在公開的推論引擎設計中結合連續批次與 KV Cache 管理，使多請求 LLM 服務能動態排程。實際效能依工作負載、硬體和配置而異。",
    related: ["kv-cache", "paged-attention", "latency-throughput", "model-serving"],
    sources: [
      ["vLLM documentation", "https://docs.vllm.ai/en/latest/"],
      [
        "Orca paper",
        "https://www.usenix.org/conference/osdi22/presentation/yu",
      ],
    ],
  },
  {
    id: "paged-attention",
    title: "PagedAttention",
    zh: "分頁式注意力快取管理",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary: "將 KV Cache 分配為較小的 blocks，由映射表管理非連續實體記憶體。",
    intuition: "像作業系統的分頁記憶體：邏輯上連續，實際上可以分散存放。",
    mechanism: [
      "長度不固定的 request 讓連續 KV Cache 配置容易產生內部/外部碎片與預留浪費。",
      "PagedAttention 將序列的 KV 儲存切分為 blocks，藉由 block table 將邏輯 token 區段對應到實體 block。",
      "Scheduler 可依需求配置和釋放 blocks，也可在特定場景共用 blocks，降低部分重複儲存。",
      "代價包括更複雜的 kernel addressing、block management 與不同形狀工作負載下的權衡。",
    ],
    diagram: [
      "Sequence token positions",
      "Logical KV blocks",
      "Block mapping table",
      "Physical GPU memory blocks",
      "Paged attention lookup",
    ],
    code: `# Conceptual mapping, not a runnable vLLM kernel
block_table = {0: 7, 1: 2, 2: 15}
logical_block = token_index // block_size
physical_block = block_table[logical_block]
offset = token_index % block_size`,
    caseTitle: "vLLM — PagedAttention",
    case: "vLLM 的公開論文詳細說明分頁式 KV 管理如何減少記憶體浪費，並提高可同時處理的請求量。",
    related: ["kv-cache", "continuous-batching", "gpu-memory"],
    sources: [
      ["vLLM paper", "https://arxiv.org/abs/2309.06180"],
      ["vLLM official docs", "https://docs.vllm.ai/en/latest/"],
    ],
  },
  {
    id: "operator-fusion",
    title: "Operator Fusion",
    zh: "算子融合",
    category: "Compiler Engineering",
    level: "Core focus",
    status: "published",
    summary: "把可兼容的運算合併，以減少中間記憶體存取與啟動開銷。",
    intuition: "原本每一道工序都要把材料搬進搬出倉庫；融合後可以在同一個工作站直接做完。",
    mechanism: [
      "以 elementwise chain 為例，獨立 kernels 可能將每個中間 Tensor 寫回 global memory，再由下一個 kernel 讀取。",
      "Compiler 會檢查運算依賴、形狀、layout、合法性與硬體限制，將可融合操作映射到同一 kernel。",
      "Fusion 能降低 kernel launches 和中間資料搬運，卻可能增加 register pressure、降低 occupancy，或改變浮點數運算順序。",
      "對 GEMM + epilogue 的融合，要區分大型矩陣核心運算與 bias/activation 等後處理；不能假設任意 operator 都能直接融合。",
    ],
    diagram: [
      "Graph: MatMul → Bias → ReLU",
      "Compiler pattern match",
      "Legality + cost checks",
      "Fused kernel / epilogue",
      "Benchmark correctness + latency",
    ],
    code: `import torch
class Block(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.relu(x @ weight + bias)

# On a supported backend, inspect generated code.
# compiled = torch.compile(Block())`,
    caseTitle: "PyTorch — TorchInductor",
    case: "PyTorch 的 torch.compile 透過 TorchDynamo/AOTAutograd/Inductor 等元件擷取與最佳化運算，並可能將適當的算子融合成更有效的執行方式。需透過 benchmark 驗證是否加速。",
    related: ["torch-compile", "computation-graph", "triton", "gpu-memory"],
    sources: [
      ["PyTorch torch.compile", "https://docs.pytorch.org/docs/stable/generated/torch.compile.html"],
      [
        "TorchInductor overview",
        "https://pytorch.org/docs/stable/torch.compiler.html",
      ],
    ],
  },
  {
    id: "torch-compile",
    title: "torch.compile",
    zh: "PyTorch 模型編譯",
    category: "Compiler Engineering",
    level: "Core focus",
    status: "published",
    summary:
      "將符合支援條件的 PyTorch 執行路徑編譯與最佳化，而非單純切換到更快的 eager mode。",
    intuition: "像把重複執行的料理流程事先整理成更有效率的工序，但規劃本身也需要時間。",
    mechanism: [
      "TorchDynamo 擷取 Python 執行中的運算圖，並透過 guards 保護其適用假設。",
      "後續編譯流程依 backend 處理，常見預設 backend 為 Inductor，可進行 operator fusion 與程式碼產生。",
      "Graph breaks 會使部分操作回到 Python eager，dynamic shapes 可能引發 recompilation。",
      "比較效能時要區分 compilation latency 與 steady-state latency，並測試輸出正確性及不同 shapes。",
    ],
    diagram: [
      "Python PyTorch model",
      "TorchDynamo capture + guards",
      "FX graph / transformations",
      "Inductor code generation",
      "Compiled execution",
    ],
    code: `import torch
model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU())
compiled = torch.compile(model)  # supported backend required
x = torch.randn(32, 64)
print(torch.allclose(model(x), compiled(x), atol=1e-5))`,
    caseTitle: "PyTorch — Production-oriented compiler stack",
    case: "PyTorch 官方將 torch.compile 作為框架內可選的模型最佳化入口，供工程師測試既有模型的最佳化收益。特定模型是否變快，必須實際測量。",
    related: ["operator-fusion", "computation-graph", "triton", "latency-throughput"],
    sources: [
      ["torch.compile tutorial", "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html"],
      [
        "PyTorch compiler FAQ",
        "https://docs.pytorch.org/docs/stable/torch.compiler_faq.html",
      ],
    ],
  },
  {
    id: "triton",
    title: "Triton",
    zh: "GPU Kernel 程式語言與編譯器",
    category: "GPU Computing",
    level: "Core focus",
    status: "published",
    summary:
      "使用 Python-like 語法描述 GPU tile 級的平行運算，編譯成裝置執行的 kernels。",
    intuition: "不用逐一指揮每位工人，而是告訴每一小組如何處理一塊資料。",
    mechanism: [
      "Triton 以 program instance 與 block/tile 的角度描述 GPU 運算，使用 tl.load/tl.store 與 mask 控制資料存取。",
      "Kernel 的 tile size、layout 與運算模式直接影響 memory coalescing、occupancy、register 壓力與效能。",
      "它不等於純 Python 在 CPU 上計算，也不能直接把 Triton JIT kernels 當作 Apple MPS CUDA kernels 使用。",
      "常見案例包括 vector add、softmax、matmul 與 fused attention kernels。",
    ],
    diagram: [
      "Python host launch",
      "@triton.jit kernel",
      "Program instances / tiles",
      "GPU memory load + compute",
      "GPU memory store",
    ],
    code: `import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    a = tl.load(X + idx, mask=mask, other=0)
    b = tl.load(Y + idx, mask=mask, other=0)
    tl.store(OUT + idx, a + b, mask=mask)
# Requires supported GPU and host-side launch code.`,
    caseTitle: "PyTorch Inductor — Triton-generated GPU kernels",
    case: "PyTorch Inductor 可使用 Triton 生成 GPU kernels，支援部分融合與最佳化的計算路徑。實際部署會依 framework 和 backend 版本而不同。",
    related: ["operator-fusion", "torch-compile", "gpu-memory", "cuda"],
    sources: [
      ["Triton tutorials", "https://triton-lang.org/main/getting-started/tutorials/"],
      ["Triton official docs", "https://triton-lang.org/main/"],
    ],
  },
  {
    id: "gpu-memory",
    title: "GPU Memory Hierarchy",
    zh: "GPU 記憶體階層",
    category: "GPU Computing",
    level: "Reference",
    status: "published",
    summary:
      "理解 registers、shared memory、cache 與 global memory 的容量、頻寬、延遲及存取規則。",
    intuition:
      "不同大小與速度的工作桌、工具櫃和倉庫；擺放資料的位置會影響整體效率。",
    mechanism: [
      "Registers 是 thread 的快速暫存資源；shared memory 可供同一 block 中的 threads 協同使用。",
      "Global memory 可容納大型 tensors，但資料存取形態與 coalescing 會影響有效頻寬。",
      "Occupancy 取決於 register、shared memory、thread/block 配置等限制，不等於越高效能一定越好。",
    ],
    diagram: ["GPU kernel threads", "Registers / shared memory", "Caches", "Global memory (VRAM)"],
    code: `// CUDA concept: cooperative tile loading
// __shared__ float tile[16][16];
// tile[threadIdx.y][threadIdx.x] = input[index];
// __syncthreads();`,
    caseTitle: "NVIDIA — CUDA Matrix Multiplication",
    case: "NVIDIA 的 CUDA Programming Guide 以 memory hierarchy 與 tiling 說明如何減少 global memory 存取；這是 GPU kernel 最佳化的基礎。",
    related: ["cuda", "triton", "operator-fusion", "kv-cache"],
    sources: [
      [
        "NVIDIA CUDA programming guide",
        "https://docs.nvidia.com/cuda/cuda-programming-guide/",
      ],
    ],
  },
  {
    id: "attention",
    title: "Attention",
    zh: "注意力機制",
    category: "Deep Learning",
    level: "Reference",
    status: "published",
    summary: "利用 Query、Key、Value 計算 token 之間的依存關係。",
    intuition:
      "處理目前這個 token 時，決定過去哪些 token 的資訊最值得參考。",
    mechanism: [
      "對 QKᵀ 進行縮放與 softmax，得到權重，再加權聚合 V。",
      "因果遮罩防止 decoder 的 token 看到未來位置。",
      "Attention 的運算與記憶體開銷，直接影響長上下文 LLM 的訓練和推論。",
    ],
    diagram: [
      "Input embeddings",
      "Q / K / V projections",
      "Scaled dot-product + mask",
      "Softmax attention weights",
      "Weighted sum of V",
    ],
    code: `import torch
q = torch.randn(1, 4, 8)
k = torch.randn(1, 4, 8)
v = torch.randn(1, 4, 8)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`,
    caseTitle: "PyTorch — Scaled Dot Product Attention",
    case: "PyTorch 公開提供 scaled_dot_product_attention API 與後端最佳化實作，讓模型利用支援的加速路徑。",
    related: ["kv-cache", "paged-attention", "triton"],
    sources: [
      [
        "PyTorch scaled_dot_product_attention",
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
      ],
    ],
  },
  {
    id: "computation-graph",
    title: "Computation Graph / IR",
    zh: "計算圖與中介表示",
    category: "Compiler Engineering",
    level: "Reference",
    status: "published",
    summary:
      "以節點與依賴邊表示運算，讓編譯器能分析、重新安排並最佳化模型。",
    intuition:
      "先把程式攤平成流程圖，才有機會辨識重複、可以合併、可以提前計算的步驟。",
    mechanism: [
      "計算圖表達 tensor operators、資料依賴與形狀等資訊。",
      "Intermediate Representation (IR) 讓優化 passes 在結構化表示上處理 pattern matching、fusion 與 code generation。",
      "Graph break 與動態控制流程可能阻止某些編譯最佳化。",
    ],
    diagram: [
      "Model / Python frontend",
      "Graph capture",
      "Intermediate representation",
      "Optimization passes",
      "Backend codegen",
    ],
    code: `import torch
from torch.fx import symbolic_trace
model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
print(symbolic_trace(model).graph)`,
    caseTitle: "PyTorch — FX graph transformations",
    case: "PyTorch FX 提供可檢視和轉換的圖表示方式，供圖層級分析與工具建置使用。",
    related: ["torch-compile", "operator-fusion"],
    sources: [
      [
        "PyTorch FX documentation",
        "https://docs.pytorch.org/docs/stable/fx.html",
      ],
    ],
  },
  {
    id: "latency-throughput",
    title: "Latency vs Throughput",
    zh: "延遲與吞吐量",
    category: "Inference Optimization",
    level: "Reference",
    status: "published",
    summary:
      "Latency 關注每個請求多久完成；throughput 關注單位時間能處理多少工作。",
    intuition:
      "一輛車多久到目的地，和整條道路每小時能通過多少輛車，是兩個不同指標。",
    mechanism: [
      "LLM 服務常測量 TTFT、TPOT 或 inter-token latency、end-to-end latency、tokens/s。",
      "Batching 提高 GPU 利用率時，可能增加排隊時間與單請求延遲。",
      "Benchmark 必須說明併發、輸入/輸出長度、暖機和 p50/p95/p99。",
    ],
    diagram: [
      "Request arrival",
      "Queue wait",
      "Prefill: first token",
      "Decode tokens",
      "Request completed",
    ],
    code: `latencies_ms = [42, 51, 57, 110]
latencies_ms.sort()
print('sample median ms:', (latencies_ms[1]+latencies_ms[2])/2)
# Real benchmarks require much larger samples.`,
    caseTitle: "NVIDIA Triton Inference Server — Performance Analyzer",
    case: "NVIDIA Triton Inference Server 提供效能分析工具，協助評估推論延遲和吞吐量，並依實際服務工作負載調整設定。",
    related: ["continuous-batching", "model-serving", "kv-cache"],
    sources: [
      [
        "Triton Inference Server docs",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/",
      ],
    ],
  },
  {
    id: "cuda",
    title: "CUDA",
    zh: "NVIDIA GPU 平行運算平台",
    category: "GPU Computing",
    level: "Glossary",
    status: "glossary",
    summary:
      "NVIDIA GPU 運算平台與程式設計模型，涵蓋 kernel、block、thread、memory 與 streams。",
    related: ["triton", "gpu-memory", "operator-fusion"],
    sources: [
      ["CUDA guide", "https://docs.nvidia.com/cuda/cuda-programming-guide/"],
    ],
  },
  {
    id: "model-serving",
    title: "Model Serving",
    zh: "模型線上服務",
    category: "Data & MLOps",
    level: "Glossary",
    status: "glossary",
    summary:
      "把模型部署成可靠、可擴展、可量測的線上服務，處理併發、版本、延遲與失敗。",
    related: ["latency-throughput", "continuous-batching"],
    sources: [
      [
        "NVIDIA Triton Inference Server",
        "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/",
      ],
    ],
  },
  {
    id: "ddp",
    title: "Distributed Data Parallel",
    zh: "分散式資料平行",
    category: "Distributed Systems",
    level: "Glossary",
    status: "glossary",
    summary:
      "不同 worker 處理不同資料切片並同步梯度，提高訓練的總處理量。",
    related: ["all-reduce", "gpu-memory"],
    sources: [
      [
        "PyTorch DDP",
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html",
      ],
    ],
  },
  {
    id: "all-reduce",
    title: "All-Reduce",
    zh: "全域歸約",
    category: "Distributed Systems",
    level: "Glossary",
    status: "glossary",
    summary:
      "跨 worker 彙總數值並將結果分發給所有參與者，常用於梯度同步。",
    related: ["ddp"],
    sources: [
      [
        "PyTorch distributed",
        "https://docs.pytorch.org/docs/stable/distributed.html",
      ],
    ],
  },
  {
    id: "python",
    title: "Python",
    zh: "Python 程式語言",
    category: "Programming",
    level: "Glossary",
    status: "glossary",
    summary: "AI 原型、資料處理、框架 API 與系統協作常用語言。",
    related: ["attention", "torch-compile"],
    sources: [["Python documentation", "https://docs.python.org/3/"]],
  },
  {
    id: "cpp",
    title: "Modern C++",
    zh: "現代 C++",
    category: "Programming",
    level: "Glossary",
    status: "glossary",
    summary: "高效能系統、推論引擎與 GPU host-side 開發的重要語言。",
    related: ["cuda", "model-serving"],
    sources: [["C++ reference", "https://isocpp.org/"]],
  },
  {
    id: "rag",
    title: "Retrieval-Augmented Generation",
    zh: "檢索增強生成",
    category: "LLM & Agents",
    level: "Glossary",
    status: "glossary",
    summary: "先檢索外部資料，再將檢索結果提供給語言模型作答。",
    related: ["attention", "model-serving"],
    sources: [
      [
        "Hugging Face RAG",
        "https://huggingface.co/docs/transformers/model_doc/rag",
      ],
    ],
  },
  {
    id: "ml-evaluation",
    title: "ML Evaluation",
    zh: "機器學習評估",
    category: "Machine Learning",
    level: "Glossary",
    status: "glossary",
    summary:
      "以合適的切分、指標、基準線與錯誤分析衡量模型能力。",
    related: ["latency-throughput"],
    sources: [
      [
        "scikit-learn model evaluation",
        "https://scikit-learn.org/stable/modules/model_evaluation.html",
      ],
    ],
  },
];
