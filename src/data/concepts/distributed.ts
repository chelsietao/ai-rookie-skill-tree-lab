import type { Concept } from "@/types";

export const distributedConcepts: Concept[] = [
  {
    id: "ddp",
    title: "Distributed Data Parallel",
    zh: "分散式資料平行",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "每個 worker 持有完整模型副本、處理不同的資料切片，並在反向傳播時以 All-Reduce 同步梯度，提高訓練吞吐量。",
    intuition:
      "像多位老師各自批改不同學生的考卷，批完後把大家的心得平均起來，再讓每位老師都用同一份更新後的評分標準。",
    mechanism: [
      "每個 process（通常對應一張 GPU）建立相同的模型副本，起始權重一致，並使用 DistributedSampler 讓各自讀取不同資料。",
      "前向與反向在各自資料上獨立進行，各 worker 得到各自的梯度。",
      "DDP 把參數分成 bucket，在某個 bucket 的梯度都算好時就啟動 All-Reduce，使通訊與後續的反向計算重疊。",
      "同步後每個 worker 的梯度相同，各自以相同的優化器更新，權重因此保持一致。每張 GPU 仍需放得下完整模型、梯度與優化器狀態。",
    ],
    diagram: [
      "每個 rank 持有完整模型副本",
      "各自處理不同 mini-batch",
      "反向傳播產生梯度",
      "All-Reduce 平均梯度",
      "各自更新參數",
    ],
    code: `import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
model = torch.nn.Linear(10, 1).to(rank)
ddp_model = DDP(model, device_ids=[rank])
# 以 torchrun --nproc_per_node=N train.py 啟動`,
    caseTitle: "PyTorch DDP 設計論文",
    case: "PyTorch 團隊發表論文說明 DDP 的設計，包含梯度分桶與通訊計算重疊，並報告了擴展性結果。DDP 適用於模型能放進單卡的情況，更大的模型常需要 FSDP 或模型平行。",
    related: ["all-reduce", "fsdp", "nccl", "gpu-memory"],
    sources: [
      ["PyTorch DDP", "https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"],
      ["PyTorch Distributed: Experiences on Accelerating Data Parallel Training", "https://arxiv.org/abs/2006.15704"],
    ],
  },
  {
    id: "all-reduce",
    title: "All-Reduce",
    zh: "全域歸約",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "跨多個 worker 對各自的資料做歸約（例如加總或平均），並讓每個 worker 都得到相同的結果，是梯度同步的核心集體通訊操作。",
    intuition:
      "像一桌人各自報出自己的數字，大家一起算出總和，最後每個人手上都有同一個答案。",
    mechanism: [
      "每個 rank 持有一份張量，All-Reduce 結束後，每個 rank 都得到所有張量逐元素歸約（sum、avg、max 等）的結果。",
      "Ring All-Reduce 把張量切成 N 份，先做 N−1 步的 reduce-scatter，再做 N−1 步的 all-gather；每個 rank 傳送的資料量與 N 幾乎無關，因此頻寬使用效率高。",
      "也有 tree 等演算法，在節點數多或延遲敏感時各有優勢；NCCL 會依拓樸與資料大小選擇演算法。",
      "效能取決於互連頻寬（NVLink、InfiniBand、乙太網路）與訊息大小，小訊息偏向延遲受限，大訊息偏向頻寬受限。",
    ],
    diagram: [
      "每個 rank 持有本地梯度",
      "Reduce-Scatter（各 rank 得到一段加總）",
      "All-Gather（把各段分發給所有人）",
      "每個 rank 得到完整加總",
      "除以 N 得到平均",
    ],
    code: `import torch, torch.distributed as dist

dist.init_process_group("nccl")
t = torch.ones(4, device="cuda") * (dist.get_rank() + 1)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
# 每個 rank 的 t 都是所有 rank 的逐元素總和`,
    caseTitle: "Horovod 與 NCCL",
    case: "Horovod 論文介紹了以 ring all-reduce 做深度學習梯度同步的實作；NVIDIA NCCL 文件列出 all-reduce 等集體通訊操作。實際效能取決於硬體互連與拓樸。",
    related: ["ddp", "nccl", "fsdp", "tensor-parallelism"],
    sources: [
      ["Horovod paper", "https://arxiv.org/abs/1802.05799"],
      ["NCCL collective operations", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html"],
    ],
  },
  {
    id: "fsdp",
    title: "Fully Sharded Data Parallel",
    zh: "完全分片資料平行",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "把模型參數、梯度與優化器狀態分片到所有 worker，只在需要時暫時收集完整參數，讓單卡放不下的模型也能以資料平行訓練。",
    intuition:
      "像一套很厚的百科全書，每人只保管其中一冊；要用到某一冊時，大家暫時把它傳給所有人，用完就收回。",
    mechanism: [
      "平時每個 rank 只保存自己負責的那份參數分片、梯度分片與優化器狀態分片，大幅降低每張卡的記憶體用量。",
      "計算某一層的前向時，先以 all-gather 收集該層的完整參數，算完即釋放。",
      "反向傳播時同樣先 all-gather 參數，算出梯度後用 reduce-scatter 把梯度分片分配回各 rank。",
      "用更多通訊換取更低的記憶體；可透過預取、通訊與計算重疊與分片單位的設定來調整效能。",
    ],
    diagram: [
      "參數／梯度／優化器狀態分片儲存",
      "All-gather 取得某層完整參數",
      "前向與反向計算",
      "Reduce-scatter 分配梯度分片",
      "各自更新自己的分片",
    ],
    code: `import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 假設已 init_process_group 並將 model 放在對應 GPU
model = FSDP(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)`,
    caseTitle: "PyTorch FSDP 論文",
    case: "PyTorch 團隊發表論文說明 FSDP 的設計與工業界經驗，並提供官方文件。其概念與 DeepSpeed 的 ZeRO Stage 3 相近，各自的 API 與細節不同。",
    related: ["ddp", "zero", "gpu-memory", "nccl"],
    sources: [
      ["PyTorch FSDP paper", "https://arxiv.org/abs/2304.11277"],
      ["PyTorch FSDP documentation", "https://docs.pytorch.org/docs/stable/fsdp.html"],
    ],
  },
  {
    id: "zero",
    title: "ZeRO",
    zh: "零冗餘優化器",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "把資料平行中重複保存的優化器狀態、梯度與參數逐階段分片到各 worker，消除記憶體冗餘。",
    intuition:
      "像一家分店人人都影印了整份帳本；ZeRO 改成每人只保管一部分，需要時再互相查閱，省下大量櫃子空間。",
    mechanism: [
      "傳統資料平行中，每張卡都保存完整的參數、梯度與優化器狀態（如 Adam 的動量與變異數），造成大量重複。",
      "Stage 1 只分片優化器狀態；Stage 2 再分片梯度；Stage 3 連參數也分片。",
      "分片越多，每張卡的記憶體越省，但需要的通訊也越多，Stage 3 的通訊型態與 FSDP 相近。",
      "ZeRO 可搭配 CPU 或 NVMe 卸載（offload）進一步擴充容量，但會受限於傳輸頻寬。",
    ],
    diagram: [
      "Stage 1：分片優化器狀態",
      "Stage 2：再分片梯度",
      "Stage 3：再分片參數",
      "需要時通訊收集",
      "各 rank 更新自己的分片",
    ],
    code: `# DeepSpeed 設定示意（JSON 設定檔；欄位以官方文件為準）
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "zero_optimization": {"stage": 2},
    "bf16": {"enabled": True},
}`,
    caseTitle: "ZeRO 論文與 DeepSpeed",
    case: "ZeRO 論文分析了訓練時的記憶體組成並提出分階段分片；DeepSpeed 是實作 ZeRO 的開源函式庫。具體的記憶體節省與通訊成本取決於模型與硬體。",
    related: ["fsdp", "deepspeed", "ddp", "gpu-memory"],
    sources: [
      ["ZeRO paper", "https://arxiv.org/abs/1910.02054"],
      ["DeepSpeed ZeRO tutorial", "https://www.deepspeed.ai/tutorials/zero/"],
    ],
  },
  {
    id: "tensor-parallelism",
    title: "Tensor Parallelism",
    zh: "張量平行",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "把單一層的權重矩陣切分到多張 GPU 上並行計算，再以集體通訊合併結果，讓單層太大或需要降低單卡記憶體時仍可執行。",
    intuition:
      "像一道很大的乘法題，大家各算其中一欄，最後把各自的結果拼起來或加總。",
    mechanism: [
      "Megatron-LM 對 Transformer 的 MLP 採用「欄切分再列切分」：第一個矩陣按欄切分，各 GPU 獨立計算啟動函數，第二個矩陣按列切分，最後以一次 all-reduce 合併。",
      "對注意力層則依 head 切分，每張 GPU 處理一部分 head，輸出投影後同樣以 all-reduce 合併。",
      "每個 Transformer 層的前向與反向各需要若干次 all-reduce，因此對互連頻寬很敏感，通常限制在單一節點內（NVLink）使用。",
      "在推論中，張量平行可降低單次請求的延遲與單卡記憶體，但通訊成本會限制可擴展的程度。",
    ],
    diagram: [
      "權重矩陣切成分片",
      "各 GPU 計算自己的分片",
      "All-reduce 合併部分結果",
      "進入下一層",
      "反向傳播亦需通訊",
    ],
    code: `# 概念性：欄切分線性層（2 張 GPU）
# W: [d_in, d_out] 切成 W0, W1: [d_in, d_out/2]
y0 = x @ W0     # GPU0
y1 = x @ W1     # GPU1
y = torch.cat([y0, y1], dim=-1)   # 需要 all-gather 或接著做列切分`,
    caseTitle: "Megatron-LM",
    case: "NVIDIA 的 Megatron-LM 論文公開了 Transformer 層內的張量平行切法與實驗結果，之後許多訓練與推論框架提供類似功能。",
    related: ["megatron-lm", "pipeline-parallelism", "all-reduce", "nccl"],
    sources: [
      ["Megatron-LM paper", "https://arxiv.org/abs/1909.08053"],
      ["Megatron-LM repository", "https://github.com/NVIDIA/Megatron-LM"],
    ],
  },
  {
    id: "pipeline-parallelism",
    title: "Pipeline Parallelism",
    zh: "管線平行",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "把模型依層切成多個階段，分別放在不同 GPU 上，並把一個 batch 切成 micro-batch 讓各階段像流水線一樣同時工作。",
    intuition:
      "像工廠的裝配線：每個工作站只做一道工序，產品一個接一個通過，所有工作站就能同時忙碌。",
    mechanism: [
      "把連續的若干層分配給同一個 stage，stage 之間只傳遞啟動值（前向）與其梯度（反向），通訊量相對較小，適合跨節點。",
      "若整個 batch 一次通過，後面的 stage 會閒置；GPipe 把 batch 切成 micro-batch，讓不同 stage 同時處理不同的 micro-batch。",
      "管線啟動與收尾時仍有閒置時間，稱為 pipeline bubble；增加 micro-batch 數量可降低 bubble 占比。",
      "排程策略（如 1F1B、interleaved）可在降低 bubble 與控制啟動值記憶體之間取捨。",
    ],
    diagram: [
      "模型依層切成 stage",
      "Batch 切成 micro-batch",
      "Stage 間傳遞啟動值",
      "多個 micro-batch 流水線執行",
      "反向傳播並累積梯度",
    ],
    code: `# 概念性排程（4 stage、4 micro-batch 的前向）
# 時間 t 上，stage s 處理 micro-batch (t - s)
for t in range(n_stages + n_micro - 1):
    for s in range(n_stages):
        m = t - s
        if 0 <= m < n_micro:
            run_forward(stage=s, micro_batch=m)`,
    caseTitle: "GPipe 與 PipeDream",
    case: "GPipe 與 PipeDream 論文分別提出以 micro-batch 與不同排程實現管線平行；Megatron-LM 後續工作整合了 interleaved 排程。實際效率取決於 stage 切分是否平衡與 micro-batch 數。",
    related: ["tensor-parallelism", "megatron-lm", "ddp", "gpu-memory"],
    sources: [
      ["GPipe paper", "https://arxiv.org/abs/1811.06965"],
      ["PipeDream paper", "https://arxiv.org/abs/1806.03377"],
    ],
  },
  {
    id: "nccl",
    title: "NCCL",
    zh: "NVIDIA 集體通訊函式庫",
    category: "Distributed Systems",
    level: "Core focus",
    status: "published",
    summary:
      "NVIDIA 針對多 GPU 與多節點最佳化的集體通訊函式庫，提供 all-reduce、all-gather、reduce-scatter、broadcast 等操作。",
    intuition:
      "像 GPU 之間的專業物流公司：知道哪條路（NVLink、PCIe、網路）最快，並自動規劃最有效率的運送路線。",
    mechanism: [
      "NCCL 偵測系統的拓樸（GPU 間的 NVLink、PCIe 與網卡連接），建立適合的通訊環或樹。",
      "它提供集體操作的 kernel，通訊與計算可透過 CUDA stream 重疊，並支援單機多卡與跨節點的 InfiniBand 或 RoCE。",
      "PyTorch 的 torch.distributed 在 GPU 上預設使用 NCCL 作為後端。",
      "常見的除錯手段包含設定 NCCL_DEBUG 環境變數查看拓樸與演算法選擇；網路設定錯誤常是效能低落或 hang 住的原因。",
    ],
    diagram: [
      "框架呼叫集體操作",
      "NCCL 偵測拓樸並選擇演算法",
      "GPU 間經 NVLink / 網路傳輸",
      "與計算 stream 重疊",
      "結果回到各 GPU",
    ],
    code: `# 啟動時查看 NCCL 的拓樸與演算法選擇
# NCCL_DEBUG=INFO torchrun --nproc_per_node=8 train.py`,
    caseTitle: "深度學習框架的通訊後端",
    case: "NVIDIA 公開了 NCCL 的使用指南，說明各集體操作與環境變數；PyTorch 文件也說明在 GPU 訓練時使用 NCCL 後端。拓樸與網路環境會大幅影響結果。",
    related: ["all-reduce", "ddp", "fsdp", "cuda-streams"],
    sources: [
      ["NCCL user guide", "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/"],
      ["NVIDIA NCCL", "https://developer.nvidia.com/nccl"],
    ],
  },
  {
    id: "deepspeed",
    title: "DeepSpeed",
    zh: "DeepSpeed 訓練函式庫",
    category: "Distributed Systems",
    level: "Reference",
    status: "published",
    summary:
      "微軟開源的深度學習最佳化函式庫，提供 ZeRO、卸載、混合精度與推論最佳化等功能，方便訓練與服務大型模型。",
    intuition:
      "像一套把記憶體與通訊最佳化做成設定檔的訓練加速套件：不改太多模型程式，就能用更少的記憶體訓練更大的模型。",
    mechanism: [
      "以 deepspeed.initialize 包裝模型、優化器與設定，由 JSON 設定檔指定 ZeRO 階段、精度與批次大小。",
      "ZeRO 各階段把優化器狀態、梯度與參數分片，並可選擇卸載到 CPU 或 NVMe。",
      "也提供管線平行、MoE、梯度累積與推論相關功能，具體可用功能隨版本變化。",
      "常與 Hugging Face Transformers 等訓練框架整合，但設定細節、與其他並行策略的組合方式需要依官方文件驗證。",
    ],
    diagram: [
      "模型與優化器",
      "deepspeed.initialize + 設定檔",
      "ZeRO 分片 / 卸載",
      "訓練迴圈",
      "儲存分片檢查點",
    ],
    code: `import deepspeed

engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)
loss = engine(batch)
engine.backward(loss)
engine.step()`,
    caseTitle: "開源訓練函式庫",
    case: "DeepSpeed 官方網站與 GitHub 專案公開了其功能與使用方式；ZeRO 論文是其核心技術之一。是否適合取決於模型大小、硬體與團隊既有的訓練框架。",
    related: ["zero", "fsdp", "ddp", "mixed-precision"],
    sources: [
      ["DeepSpeed documentation", "https://www.deepspeed.ai/"],
      ["DeepSpeed repository", "https://github.com/deepspeedai/DeepSpeed"],
    ],
  },
  {
    id: "megatron-lm",
    title: "Megatron-LM",
    zh: "Megatron-LM 大模型訓練框架",
    category: "Distributed Systems",
    level: "Reference",
    status: "published",
    summary:
      "NVIDIA 的大型 Transformer 訓練框架，整合張量、管線與資料平行，並針對 NVIDIA GPU 叢集最佳化。",
    intuition:
      "像一套為超大型模型準備的施工藍圖：說明怎麼把一個巨大的模型同時從「層內、層間、資料」三個方向切開，分給成千上萬張 GPU。",
    mechanism: [
      "張量平行把層內的矩陣切到同節點的多張 GPU；管線平行把不同層放在不同節點；資料平行則複製這個切分後的整體。",
      "後續工作提出 interleaved 管線排程以降低 bubble，並整合序列平行與選擇性重算等降低啟動值記憶體的方法。",
      "三種平行度的乘積等於 GPU 總數，需要依模型結構與硬體拓樸調整各自的大小。",
      "除了 Megatron-LM 本身，Megatron-Core 提供可供其他框架重用的元件，以其官方文件為準。",
    ],
    diagram: [
      "資料平行群組",
      "管線平行 stage",
      "張量平行分片",
      "NCCL 集體通訊",
      "合併梯度並更新",
    ],
    code: `# 並行度關係示意
tp, pp, dp = 8, 4, 16
world_size = tp * pp * dp     # 512 張 GPU`,
    caseTitle: "Megatron-LM 論文系列",
    case: "NVIDIA 的論文公開了張量平行與在 GPU 叢集上整合三種平行策略的設計與實驗結果。其倉庫提供程式碼與文件。",
    related: ["tensor-parallelism", "pipeline-parallelism", "ddp", "nccl"],
    sources: [
      ["Megatron-LM paper", "https://arxiv.org/abs/1909.08053"],
      ["Efficient Large-Scale Language Model Training on GPU Clusters", "https://arxiv.org/abs/2104.04473"],
    ],
  },
  {
    id: "distributed-checkpointing",
    title: "Distributed Checkpointing",
    zh: "分散式檢查點",
    category: "Distributed Systems",
    level: "Reference",
    status: "published",
    summary:
      "在多卡、多節點訓練中，以分片方式並行儲存與載入模型與優化器狀態，使訓練能從中斷處恢復，並支援不同並行配置之間的重新載入。",
    intuition:
      "像一份大型專案的進度存檔，每位成員各自存自己負責的部分，不必全部傳到同一台電腦再存；之後換人手數量也能把進度讀回來。",
    mechanism: [
      "大型訓練的參數與優化器狀態分布在許多 rank 上，若集中到單一 rank 儲存，會造成記憶體與 I/O 瓶頸。",
      "分散式檢查點讓每個 rank 並行寫出自己的分片，並記錄中繼資料描述每個張量如何切分。",
      "載入時依中繼資料與目前的並行配置重新切分，因此可以在不同 GPU 數量或平行策略下恢復（resharding）。",
      "實務上需要考慮非同步儲存以減少訓練停頓、保留多個版本以及故障後的完整性檢查。",
    ],
    diagram: [
      "訓練中的分片狀態",
      "各 rank 並行寫出分片",
      "寫入中繼資料",
      "故障或調整配置",
      "依中繼資料重新分片載入",
    ],
    code: `import torch.distributed.checkpoint as dcp

state = {"model": model.state_dict()}
dcp.save(state, checkpoint_id="ckpt/step_1000")
# 之後以 dcp.load(state, checkpoint_id=...) 載入`,
    caseTitle: "PyTorch Distributed Checkpoint",
    case: "PyTorch 提供 torch.distributed.checkpoint 模組，文件說明其以分片與中繼資料支援並行儲存與重新分片載入。具體 API 隨版本演進，使用前請對照所用版本的文件。",
    related: ["fsdp", "ddp", "zero", "model-registry"],
    sources: [
      ["PyTorch distributed checkpoint", "https://docs.pytorch.org/docs/stable/distributed.checkpoint.html"],
      ["PyTorch FSDP documentation", "https://docs.pytorch.org/docs/stable/fsdp.html"],
    ],
  },
];
