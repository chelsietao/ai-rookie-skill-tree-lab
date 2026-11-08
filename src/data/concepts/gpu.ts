import type { Concept } from "@/types";

export const gpuConcepts: Concept[] = [
  {
    id: "cuda",
    title: "CUDA",
    zh: "NVIDIA GPU 平行運算平台",
    category: "GPU Computing",
    level: "Core focus",
    status: "published",
    summary:
      "NVIDIA 的 GPU 平行運算平台與程式設計模型，以 kernel、grid、block、thread 與多層記憶體組織大量並行運算。",
    intuition:
      "像指揮一個擁有成千上萬名工人的工廠：你寫一份工作說明（kernel），並說明要分成多少個工作小組（block），每位工人憑自己的編號處理資料中的一小塊。",
    mechanism: [
      "Host（CPU）端程式啟動 kernel，並指定 grid 與 block 的維度；每個 thread 以 threadIdx、blockIdx 等內建變數得知自己要處理的資料位置。",
      "Block 內的 thread 可透過 shared memory 與同步（__syncthreads）協作；不同 block 之間彼此獨立，由硬體排程到各個 SM 上執行。",
      "記憶體分成 global、shared、register、constant 等層級，延遲與頻寬差異很大，效能調校常以資料搬移為核心。",
      "Host 與 device 之間的資料複製、kernel 啟動與同步通常以 stream 組織，才能讓傳輸與計算重疊。",
    ],
    diagram: [
      "Host 配置並複製資料到 device",
      "啟動 kernel（grid × block）",
      "SM 排程 block 與 warp",
      "Thread 讀寫各層記憶體",
      "結果複製回 host",
    ],
    code: `// 向量相加 kernel
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
// host 端：add<<<(n + 255) / 256, 256>>>(a, b, c, n);`,
    caseTitle: "深度學習框架的底層",
    case: "PyTorch 等框架在 NVIDIA GPU 上的許多運算最終由 CUDA kernel 或其上的函式庫執行；NVIDIA 公開了 CUDA 程式設計指南與函式庫文件。學習 CUDA 有助於理解效能瓶頸，但日常工作未必需要自己撰寫 kernel。",
    related: ["triton", "gpu-memory", "operator-fusion", "warp-simt"],
    sources: [
      ["CUDA programming guide", "https://docs.nvidia.com/cuda/cuda-programming-guide/"],
      ["CUDA C++ Programming Guide", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"],
    ],
  },
  {
    id: "warp-simt",
    title: "Warp and SIMT",
    zh: "Warp 與 SIMT 執行模型",
    category: "GPU Computing",
    level: "Reference",
    status: "published",
    summary:
      "NVIDIA GPU 以 32 個 thread 為一個 warp 執行，採用 SIMT（單指令多執行緒）模型，分支發散會降低效率。",
    intuition:
      "像一排 32 位舞者聽同一個口令：如果有人往左、有人往右，口令就得分兩輪下，整排的效率就掉了。",
    mechanism: [
      "Block 內的 thread 被硬體分成每 32 個一組的 warp，warp 是排程與指令發射的基本單位。",
      "同一個 warp 的 thread 在同一時刻執行同一道指令，各自帶著自己的暫存器與資料。",
      "遇到 if/else 且 warp 內 thread 走不同路徑時，硬體會依序執行各分支並遮罩不參與的 thread，稱為 warp divergence。",
      "SM 以大量常駐的 warp 掩蓋記憶體延遲；每個 thread 的暫存器與 shared memory 用量會影響可同時常駐的 warp 數（occupancy）。",
    ],
    diagram: [
      "Block 內的 thread",
      "每 32 個分成一個 warp",
      "Warp 以 SIMT 發射指令",
      "遇到分支則遮罩並序列化",
      "SM 切換其他 warp 掩蓋延遲",
    ],
    code: `// 容易造成 divergence：同一 warp 內奇偶 thread 走不同分支
__global__ void k(float* x) {
  int i = threadIdx.x;
  if (i % 2 == 0) x[i] *= 2.0f;
  else            x[i] += 1.0f;
}`,
    caseTitle: "CUDA 官方文件的 SIMT 描述",
    case: "CUDA 程式設計指南說明了 SIMT 架構與 warp 的行為，並提醒分支發散的影響。實際影響程度需用 profiler 觀察，現代 GPU 的細節（例如獨立 thread 排程）也隨架構演進。",
    related: ["cuda", "gpu-memory", "memory-coalescing", "nsight-profiler"],
    sources: [
      ["CUDA C++ Programming Guide", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"],
      ["CUDA C++ Best Practices Guide", "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"],
    ],
  },
  {
    id: "tensor-cores",
    title: "Tensor Cores",
    zh: "張量核心",
    category: "GPU Computing",
    level: "Core focus",
    status: "published",
    summary:
      "NVIDIA GPU 中專為矩陣乘加（MMA）設計的硬體單元，支援 FP16、BF16、TF32、INT8、FP8 等低精度格式，大幅提升矩陣運算吞吐。",
    intuition:
      "一般運算單元像一個個人手算乘法；Tensor Core 則像一台專門蓋整塊小矩陣章的機器，一次處理一小塊矩陣的乘加。",
    mechanism: [
      "Tensor Core 對小尺寸的矩陣片段執行 D = A×B + C，輸入常為低精度、累加常用較高精度。",
      "要用到 Tensor Core，運算需使用支援的資料型態，且矩陣維度、對齊要符合函式庫或 kernel 的要求。",
      "cuBLAS、cuDNN 與 CUTLASS 等函式庫會在條件符合時自動選用 Tensor Core 的 kernel；Triton 與 PyTorch 的混合精度（autocast）也能受益。",
      "支援的資料型態與吞吐隨 GPU 世代而異，需對照所用 GPU 的規格。",
    ],
    diagram: [
      "模型使用 FP16 / BF16 / FP8",
      "矩陣乘法切成小片段",
      "Tensor Core 執行 MMA",
      "較高精度累加",
      "輸出結果",
    ],
    code: `import torch

a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
c = a @ b   # 在支援的 GPU 上通常會選用 Tensor Core kernel`,
    caseTitle: "混合精度訓練",
    case: "NVIDIA 的效能指南說明了矩陣乘法如何受益於 Tensor Core，以及維度對齊等條件。實際加速取決於 GPU、矩陣形狀與函式庫版本。",
    related: ["mixed-precision", "quantization", "cuda", "roofline-model"],
    sources: [
      ["NVIDIA Matrix Multiplication Background", "https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html"],
      ["NVIDIA GPU Performance Background", "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html"],
    ],
  },
  {
    id: "roofline-model",
    title: "Roofline Model",
    zh: "Roofline 效能模型",
    category: "GPU Computing",
    level: "Core focus",
    status: "published",
    summary:
      "以運算強度（FLOPs/Byte）為橫軸，說明一個 kernel 的效能上限受運算峰值或記憶體頻寬哪一個限制的視覺化模型。",
    intuition:
      "像一間廚房的出餐速度，不是被廚師手速限制，就是被冰箱取料的速度限制；Roofline 幫你判斷現在卡在哪一邊。",
    mechanism: [
      "運算強度（arithmetic intensity）= 浮點運算次數 ÷ 從記憶體搬移的位元組數。",
      "可達效能上限 = min(峰值算力, 運算強度 × 記憶體頻寬)；強度低的 kernel 落在斜線區（memory-bound），強度高的落在水平線區（compute-bound）。",
      "大型矩陣乘法強度高，常屬 compute-bound；逐元素運算與 LLM 的 decode 階段強度低，常屬 memory-bound。",
      "這也解釋了 operator fusion、量化與 KV Cache 最佳化的價值：減少記憶體搬移可提升有效強度。",
    ],
    diagram: [
      "量測 FLOPs 與資料搬移量",
      "計算運算強度",
      "對照 GPU 峰值算力與頻寬",
      "判斷 memory-bound 或 compute-bound",
      "選擇對應的最佳化方向",
    ],
    code: `# 概念性計算：向量相加（FP32）
flops = n                  # 每個元素 1 次加法
bytes_moved = 3 * n * 4    # 讀 a、b，寫 c
intensity = flops / bytes_moved   # 約 1/12 FLOP/Byte，通常為 memory-bound`,
    caseTitle: "Roofline 原始論文",
    case: "Williams、Waterman 與 Patterson 在 2009 年的 CACM 論文提出 Roofline 模型；NVIDIA 的效能文件也以類似概念說明運算與記憶體瓶頸。實際分析需代入所用 GPU 的實測峰值。",
    related: ["gpu-memory", "operator-fusion", "latency-throughput", "tensor-cores"],
    sources: [
      ["Roofline: an insightful visual performance model", "https://dl.acm.org/doi/10.1145/1498765.1498785"],
      ["NVIDIA GPU Performance Background", "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html"],
    ],
  },
  {
    id: "memory-coalescing",
    title: "Memory Coalescing",
    zh: "記憶體合併存取",
    category: "GPU Computing",
    level: "Reference",
    status: "published",
    summary:
      "同一個 warp 的 thread 存取連續且對齊的 global memory 位址時，硬體可把多次存取合併為少數幾次記憶體交易。",
    intuition:
      "像一整排人同時去拿相鄰的幾本書，管理員一次就能把整排書遞出來；如果大家要的書散落在各個書架，管理員就得跑很多趟。",
    mechanism: [
      "Global memory 以固定大小的區段（transaction）為單位存取，warp 內 32 個 thread 的位址會被合併成所需的區段。",
      "若 thread i 存取位址 base + i，位址連續，通常只需要少量交易；若是大步幅（stride）或隨機存取，交易數量會增加，有效頻寬下降。",
      "資料配置（例如 row-major 與 column-major）與 thread 對應的維度會影響合併效果，常見做法是讓相鄰 thread 對應記憶體中相鄰的元素。",
      "矩陣轉置等情境可借助 shared memory 做分塊，讓全域記憶體的讀寫都保持合併。",
    ],
    diagram: [
      "Warp 內 thread 發出位址",
      "硬體檢查位址是否連續對齊",
      "合併成少量記憶體交易",
      "資料回傳至暫存器",
      "（不連續則交易數增加）",
    ],
    code: `// 合併：相鄰 thread 存取相鄰元素
__global__ void good(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}
// 不合併：以大步幅存取
__global__ void bad(const float* x, float* y, int n, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * stride < n) y[i] = x[i * stride];
}`,
    caseTitle: "CUDA 最佳實務指南",
    case: "CUDA C++ Best Practices Guide 說明了合併存取的原則與其對頻寬的影響。不同架構的合併規則細節不同，請以對應架構的文件與 profiler 為準。",
    related: ["gpu-memory", "cuda", "warp-simt", "nsight-profiler"],
    sources: [
      ["CUDA C++ Best Practices Guide", "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"],
      ["CUDA C++ Programming Guide", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"],
    ],
  },
  {
    id: "cuda-streams",
    title: "CUDA Streams",
    zh: "CUDA 串流",
    category: "GPU Computing",
    level: "Reference",
    status: "published",
    summary:
      "stream 是 GPU 上按順序執行的工作佇列；不同 stream 的工作可以併發，讓資料傳輸與計算重疊。",
    intuition:
      "像多條獨立的輸送帶：同一條帶上的工作照順序做，不同條帶上的工作則可以同時進行。",
    mechanism: [
      "同一個 stream 內的 kernel 與記憶體複製依提交順序執行；不同 stream 之間沒有保證的順序，除非以事件（event）同步。",
      "非同步的 host↔device 複製需使用 pinned（page-locked）記憶體，才能真正與 kernel 執行重疊。",
      "典型做法是把大批資料切成多份，在不同 stream 中流水線式地進行「複製、計算、複製回」。",
      "PyTorch 的 CUDA 操作預設在目前的 stream 上排隊，可用 torch.cuda.Stream 管理；跨 stream 使用張量時要特別留意同步與記憶體生命週期。",
    ],
    diagram: [
      "Stream 1：複製 → kernel → 複製回",
      "Stream 2：複製 → kernel → 複製回",
      "硬體排程重疊傳輸與計算",
      "Event 同步",
      "完成",
    ],
    code: `import torch

s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
a = torch.randn(1 << 20, device="cuda")
with torch.cuda.stream(s1):
    b = a * 2
with torch.cuda.stream(s2):
    c = a + 1
torch.cuda.synchronize()`,
    caseTitle: "計算與傳輸重疊",
    case: "CUDA 官方文件描述了以 stream 達成非同步並行執行的方法與限制。是否能獲得重疊效益取決於硬體、記憶體型態與工作量大小，需以 profiler 驗證。",
    related: ["cuda", "gpu-memory", "nsight-profiler", "latency-throughput"],
    sources: [
      ["CUDA C++ Programming Guide: Asynchronous Concurrent Execution", "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"],
      ["PyTorch CUDA semantics", "https://docs.pytorch.org/docs/stable/notes/cuda.html"],
    ],
  },
  {
    id: "nsight-profiler",
    title: "Nsight Systems and Nsight Compute",
    zh: "NVIDIA 效能分析工具",
    category: "GPU Computing",
    level: "Core focus",
    status: "published",
    summary:
      "Nsight Systems 提供系統層級的時間軸分析，Nsight Compute 針對單一 CUDA kernel 做深入的指標與瓶頸分析。",
    intuition:
      "Nsight Systems 像整座工廠的監視錄影，看每個環節何時忙、何時閒；Nsight Compute 像把一台機器拆開，量測內部每個零件的使用率。",
    mechanism: [
      "先用 Nsight Systems 看整體時間軸：CPU 執行緒、kernel 啟動、記憶體複製與 stream 並行狀況，找出 GPU 閒置與瓶頸階段。",
      "找到耗時的 kernel 後，用 Nsight Compute 蒐集該 kernel 的指標，例如記憶體頻寬使用率、算力使用率與 occupancy。",
      "搭配 Roofline 思維判斷它是 memory-bound 還是 compute-bound，決定要改善存取模式、融合運算或提高並行度。",
      "最佳化後重新量測，確認改善並避免只憑直覺調整。",
    ],
    diagram: [
      "Nsight Systems 全域時間軸",
      "找出耗時階段或 GPU 閒置",
      "Nsight Compute 分析單一 kernel",
      "判斷瓶頸類型",
      "修改後重新量測",
    ],
    code: `# 命令列示意（參數以官方文件為準）
# nsys profile -o report python train.py
# ncu --set full -o kernel_report python train.py`,
    caseTitle: "效能分析的標準流程",
    case: "NVIDIA 官方網站提供 Nsight Systems 與 Nsight Compute 的說明與文件。兩者搭配「先看整體、再看單一 kernel」是常見的分析流程，但具體指標解讀需依 GPU 架構而定。",
    related: ["roofline-model", "cuda", "latency-throughput", "memory-coalescing"],
    sources: [
      ["NVIDIA Nsight Systems", "https://developer.nvidia.com/nsight-systems"],
      ["NVIDIA Nsight Compute", "https://developer.nvidia.com/nsight-compute"],
    ],
  },
];
