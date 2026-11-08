import type { Concept } from "@/types";

export const inferenceConcepts: Concept[] = [
  {
    id: "flash-attention",
    title: "FlashAttention",
    zh: "IO 感知的快速注意力",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "以分塊（tiling）在 GPU 的 SRAM 內完成精確的 Attention 計算，避免把 N×N 的分數矩陣寫回 HBM，降低記憶體流量。",
    intuition:
      "標準 Attention 像每算一步就把中間結果搬回倉庫再取出來；FlashAttention 則在手邊的工作台（SRAM）上一小塊一小塊做完，只把最終結果放回倉庫。它的輸出與標準 Attention 數學上相同，不是近似演算法。",
    mechanism: [
      "標準實作會先算出 N×N 的分數矩陣 S = QKᵀ 並寫入 HBM，再做 softmax、再與 V 相乘，序列越長，讀寫 HBM 的成本越明顯。",
      "FlashAttention 把 Q、K、V 切成小塊載入 SRAM，逐塊累積輸出，使用 online softmax 維護每列的最大值與分母，因此不需要看到完整一列分數就能得到正確的 softmax。",
      "反向傳播時不保存大型注意力矩陣，而是只保存必要的統計量，需要時再重新計算，用額外的計算換取更少的記憶體讀寫。",
      "效益主要來自減少 HBM 存取而非減少 FLOPs；實際加速取決於序列長度、head 維度、GPU 世代與 kernel 版本。",
    ],
    diagram: [
      "Q / K / V 位於 HBM",
      "分塊載入 SRAM",
      "區塊內計算分數與 online softmax",
      "累積輸出",
      "寫回 HBM（不寫出 N×N 矩陣）",
    ],
    code: `# PyTorch 的 SDPA 可依條件選用 FlashAttention 類的 kernel
import torch
import torch.nn.functional as F

q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`,
    caseTitle: "PyTorch SDPA 後端",
    case: "PyTorch 的 scaled_dot_product_attention 文件說明，函式會依輸入與硬體條件在多個實作之間選擇，其中包含 FlashAttention 類的後端。是否實際使用哪個後端取決於 dtype、形狀與 GPU，需要自行以 profiler 確認。",
    related: ["attention", "gpu-memory", "kv-cache", "triton"],
    sources: [
      ["FlashAttention paper", "https://arxiv.org/abs/2205.14135"],
      ["FlashAttention-2 paper", "https://arxiv.org/abs/2307.08691"],
    ],
  },
  {
    id: "speculative-decoding",
    title: "Speculative Decoding",
    zh: "推測式解碼",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "先用較小或較快的草稿模型一次提出多個候選 token，再由目標模型以一次前向傳播驗證，降低每個 token 的平均解碼延遲。",
    intuition:
      "像資深編輯不必逐字寫稿：由實習生先擬好一段，編輯一次看完，對的保留、從第一個錯處開始改寫。只要實習生常常猜對，整體就比編輯自己逐字寫更快。",
    mechanism: [
      "草稿機制（小模型、同模型的額外 head 或 n-gram 查詢等）一次產生 k 個候選 token。",
      "目標模型對這 k 個位置做一次前向傳播，得到各位置的機率分佈，這一步與生成一個 token 的成本相近，因為 decode 階段多半受記憶體頻寬限制。",
      "以 rejection sampling 的方式決定接受多少個候選 token，遇到第一個被拒絕的位置就由目標分佈重新取樣；在論文的設計下，輸出分佈與單獨使用目標模型取樣一致。",
      "加速幅度取決於接受率與草稿成本；接受率低或批次很大、已接近算力上限時，效益可能下降甚至為負。",
    ],
    diagram: [
      "草稿模型提出 k 個 token",
      "目標模型一次驗證 k 個位置",
      "Rejection sampling 決定接受長度",
      "保留已接受 token，必要時補一個新 token",
      "進入下一輪",
    ],
    code: `# 概念性偽碼
draft = draft_model.propose(prefix, k=4)
probs = target_model.forward(prefix + draft)   # 一次驗證
n_accept = accept_by_rejection_sampling(draft, probs)
prefix += draft[:n_accept] + [sample_next(probs, n_accept)]`,
    caseTitle: "兩篇獨立提出的論文",
    case: "Leviathan 等人與 DeepMind 的 Chen 等人各自發表了推測式解碼／取樣的演算法，論文都報告了在不改變輸出分佈的前提下的實測加速。實際收益要依模型組合、工作負載與服務框架重新量測。",
    related: ["kv-cache", "latency-throughput", "sampling-decoding", "continuous-batching"],
    sources: [
      ["Fast Inference from Transformers via Speculative Decoding", "https://arxiv.org/abs/2211.17192"],
      ["Accelerating LLM Decoding with Speculative Sampling", "https://arxiv.org/abs/2302.01318"],
    ],
  },
  {
    id: "quantization",
    title: "Quantization",
    zh: "量化",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "把權重或啟動值從 FP16/BF16 等高精度表示轉成 INT8、INT4、FP8 等低位元表示，以減少記憶體與頻寬需求。",
    intuition:
      "像把高解析度照片壓成較低解析度：檔案變小、傳輸變快，但要小心壓縮後的品質掉在可接受的範圍內。",
    mechanism: [
      "基本做法是以縮放係數（scale）與零點（zero-point）把浮點數線性映射到整數範圍，推論時再反量化或直接用低精度 kernel 計算。",
      "Post-training quantization（PTQ）在訓練後以校準資料決定縮放；quantization-aware training（QAT）則在訓練中模擬量化誤差。",
      "LLM 常用 weight-only 量化（例如 GPTQ、AWQ）降低權重記憶體；也有同時量化權重與啟動值的方案，需要有對應的低精度 kernel 才能轉化為速度。",
      "收益取決於硬體是否支援該資料型態與 kernel；品質需用任務指標與 perplexity 等評估，離群值（outliers）是常見的誤差來源。",
    ],
    diagram: [
      "FP16/BF16 權重",
      "校準資料決定 scale / zero-point",
      "轉為 INT8 / INT4 / FP8",
      "低精度 kernel 或反量化後計算",
      "評估品質與速度",
    ],
    code: `# 概念性：對稱式 INT8 量化
import torch
w = torch.randn(4096, 4096)
scale = w.abs().max() / 127
w_q = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)
w_dq = w_q.float() * scale            # 反量化
err = (w - w_dq).abs().mean()`,
    caseTitle: "GPTQ 與 AWQ",
    case: "GPTQ 與 AWQ 都是公開發表的 LLM 權重量化方法，論文報告了在 3–4 bit 權重下仍能維持接近原模型的品質；Hugging Face Transformers 文件也整理了其支援的多種量化方式。哪一種合適取決於模型、硬體與部署框架。",
    related: ["gpu-memory", "kv-cache", "latency-throughput", "mixed-precision"],
    sources: [
      ["GPTQ paper", "https://arxiv.org/abs/2210.17323"],
      ["AWQ paper", "https://arxiv.org/abs/2306.00978"],
      ["Hugging Face quantization overview", "https://huggingface.co/docs/transformers/main/en/quantization/overview"],
    ],
  },
  {
    id: "prefix-caching",
    title: "Prefix Caching",
    zh: "前綴快取",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "當多個請求共享相同的 prompt 前綴時，重複使用已計算好的 KV Cache，省下重複的 prefill 計算。",
    intuition:
      "像很多人都從同一份教材的前十頁開始問問題；與其每個人都重新讀一遍，不如把讀完前十頁的筆記共用。",
    mechanism: [
      "把 prompt 切成固定大小的 block，並以前綴內容（與先前 block）為依據計算雜湊，作為快取查找的鍵。",
      "新請求進來時，找出最長相符的已快取前綴，直接掛接對應的 KV block，只對剩下的 token 做 prefill。",
      "快取容量有限，需要 eviction 策略（例如 LRU 類）；被多個請求引用的 block 要有引用計數，避免被誤回收。",
      "常見受益情境包括固定的 system prompt、few-shot 範例、多輪對話與共享長文件；若請求前綴幾乎不重疊，效益有限。",
    ],
    diagram: [
      "新請求的 prompt",
      "分 block 並計算前綴雜湊",
      "命中則重用 KV block",
      "只 prefill 未命中的部分",
      "開始 decode",
    ],
    code: `# vLLM 以參數啟用自動前綴快取（以官方文件為準）
from vllm import LLM
llm = LLM(model="your-model", enable_prefix_caching=True)`,
    caseTitle: "vLLM Automatic Prefix Caching 與 SGLang RadixAttention",
    case: "vLLM 文件說明了自動前綴快取，SGLang 論文提出以 radix tree 管理 KV Cache 以便跨請求共享。兩者都是公開的設計，適用情境與效益需依自己的流量特性實測。",
    related: ["kv-cache", "paged-attention", "continuous-batching", "vllm"],
    sources: [
      ["vLLM Automatic Prefix Caching", "https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html"],
      ["SGLang / RadixAttention paper", "https://arxiv.org/abs/2312.07104"],
    ],
  },
  {
    id: "chunked-prefill",
    title: "Chunked Prefill",
    zh: "分塊 Prefill",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "把長 prompt 的 prefill 切成較小的區塊，與進行中的 decode 請求混合在同一個批次，降低 decode 被長 prefill 阻塞的情況。",
    intuition:
      "像大客車排隊上車時，一位帶了很多行李的乘客會擋住後面所有人；把行李分幾趟搬，後面的人就不會等太久。",
    mechanism: [
      "Prefill 階段一次處理整段 prompt，算力密集；decode 階段每步只產生一個 token，受記憶體頻寬限制，兩者的資源特性不同。",
      "長 prompt 的 prefill 若獨佔一次迭代，同批次中的 decode 請求會被拖慢，使 token 間延遲出現尖峰。",
      "Chunked prefill 設定每次迭代的 token 預算，把 prefill 切塊並與 decode token 一起排程，讓 GPU 使用率較平滑。",
      "代價是長 prompt 的首 token 時間（TTFT）可能拉長；區塊大小需要在 TTFT 與 token 間延遲之間取捨。",
    ],
    diagram: [
      "長 prompt 進入佇列",
      "依 token 預算切成多個 chunk",
      "與 decode token 混合成批次",
      "逐 chunk 完成 prefill",
      "轉入 decode",
    ],
    code: `# 概念性排程迴圈
budget = MAX_TOKENS_PER_ITER
batch = [r.next_token() for r in decoding_requests]
budget -= len(batch)
for r in prefilling_requests:
    chunk = r.take_prefill_chunk(min(budget, CHUNK))
    batch.append(chunk); budget -= len(chunk)`,
    caseTitle: "SARATHI 與 Sarathi-Serve",
    case: "SARATHI 論文提出 chunked-prefill 搭配 decode 混合批次，Sarathi-Serve 進一步討論在吞吐與延遲之間取得平衡。vLLM 等服務框架也提供相關選項，細節以其文件為準。",
    related: ["continuous-batching", "latency-throughput", "kv-cache", "prefill-decode-disaggregation"],
    sources: [
      ["SARATHI paper", "https://arxiv.org/abs/2308.16369"],
      ["Sarathi-Serve paper", "https://arxiv.org/abs/2403.02310"],
    ],
  },
  {
    id: "grouped-query-attention",
    title: "Grouped-Query Attention",
    zh: "分組查詢注意力",
    category: "Inference Optimization",
    level: "Reference",
    status: "published",
    summary:
      "讓多個 Query head 共用一組 Key/Value head，縮小 KV Cache 與解碼時的記憶體讀取量。",
    intuition:
      "像一個小組共用一份參考資料，而不是每個人各自影印一份：資料量大幅減少，但組內成員看到的是同一份。",
    mechanism: [
      "Multi-Head Attention（MHA）中每個 head 都有自己的 K、V；Multi-Query Attention（MQA）讓所有 Query head 共用一組 K、V。",
      "GQA 介於兩者之間：把 Query head 分成 G 組，每組共用一組 K、V，G 等於 1 時為 MQA，等於 head 數時為 MHA。",
      "KV Cache 大小與 KV head 數成正比，因此 GQA 能依比例降低快取記憶體與 decode 時需讀取的資料量。",
      "論文報告 GQA 的品質接近 MHA、速度接近 MQA，並說明可由既有 MHA 檢查點經少量額外訓練轉換而來。",
    ],
    diagram: [
      "Query heads（多個）",
      "分成 G 組",
      "每組對應一組 K/V head",
      "較小的 KV Cache",
      "Attention 輸出",
    ],
    code: `# 概念性：8 個 Q head、2 個 KV head（G=2）
n_q, n_kv = 8, 2
group = n_q // n_kv
k = k.repeat_interleave(group, dim=1)   # [B, n_q, T, D]
v = v.repeat_interleave(group, dim=1)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)`,
    caseTitle: "公開論文的比較",
    case: "GQA 論文比較了 MHA、MQA 與 GQA 在品質與推論速度上的取捨；MQA 則出自 Shazeer 的「One Write-Head is All You Need」。實際使用的模型是否採用 GQA，請以各模型的技術報告或設定檔為準。",
    related: ["kv-cache", "attention", "transformer", "gpu-memory"],
    sources: [
      ["GQA paper", "https://arxiv.org/abs/2305.13245"],
      ["Multi-Query Attention paper", "https://arxiv.org/abs/1911.02150"],
    ],
  },
  {
    id: "prefill-decode-disaggregation",
    title: "Prefill-Decode Disaggregation",
    zh: "Prefill／Decode 分離部署",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "把 prefill 與 decode 放在不同的 GPU 或節點群上執行，各自針對 TTFT 與 token 間延遲最佳化，並在兩者之間傳輸 KV Cache。",
    intuition:
      "像餐廳把備料廚房和出餐窗口分開：備料需要大量工作台面（算力），出餐需要快速的動線（頻寬與延遲），分開配置人力比混在一起更好調整。",
    mechanism: [
      "Prefill 是算力密集且影響 TTFT；decode 受記憶體頻寬限制且影響 token 間延遲，兩者混在同一組 GPU 會互相干擾。",
      "分離部署把請求先送到 prefill 節點，算完後把產生的 KV Cache 傳到 decode 節點繼續生成。",
      "兩種節點可以用不同的並行策略與資源比例，依 TTFT 與 TPOT 的服務目標獨立擴縮。",
      "代價是需要在節點之間高速傳輸 KV Cache，並增加排程與故障處理的複雜度；是否划算取決於網路頻寬、序列長度與流量型態。",
    ],
    diagram: [
      "請求進入",
      "Prefill 節點計算 KV Cache",
      "KV Cache 傳輸",
      "Decode 節點逐 token 生成",
      "回傳結果",
    ],
    code: `# 概念性流程
kv = prefill_pool.run(prompt)           # 算力密集
handle = transfer(kv, to=decode_pool)   # 需要高速互連
for tok in decode_pool.generate(handle):
    stream(tok)`,
    caseTitle: "DistServe 與 Splitwise",
    case: "DistServe 與 Splitwise 兩篇論文都提出把 prefill 與 decode 分開部署，並以實驗說明在特定服務目標下的收益。這些結果來自各自的實驗設定，不保證適用於所有工作負載。",
    related: ["chunked-prefill", "kv-cache", "model-serving", "latency-throughput"],
    sources: [
      ["DistServe paper", "https://arxiv.org/abs/2401.09670"],
      ["Splitwise paper", "https://arxiv.org/abs/2311.18677"],
    ],
  },
  {
    id: "sampling-decoding",
    title: "Sampling and Decoding Strategies",
    zh: "取樣與解碼策略",
    category: "Inference Optimization",
    level: "Reference",
    status: "published",
    summary:
      "決定如何從模型輸出的機率分佈中挑出下一個 token，常見方法包含 greedy、temperature、top-k、top-p 與 beam search。",
    intuition:
      "像從抽獎箱抽籤：greedy 永遠只拿機率最高的那張；temperature 調整箱內各籤的懸殊程度；top-k 與 top-p 則先把機率太低的籤拿掉再抽。",
    mechanism: [
      "模型最後一層輸出 logits，經 softmax 成為下一個 token 的機率分佈。",
      "Temperature 在 softmax 前對 logits 做縮放：值小使分佈更尖銳、輸出更確定；值大使分佈更平坦、輸出更多樣。",
      "Top-k 只保留機率最高的 k 個 token；top-p（nucleus sampling）保留累積機率達到 p 的最小集合，再於其中重新正規化並取樣。",
      "Beam search 同時保留多條候選序列，較常用於翻譯等任務；開放式文字生成常用取樣方法，因為純粹追求高機率可能導致重複、無趣的輸出。",
    ],
    diagram: [
      "Logits",
      "Temperature 縮放",
      "Top-k / Top-p 過濾",
      "Softmax 並取樣",
      "輸出 token",
    ],
    code: `import torch

def sample(logits, temperature=0.8, top_p=0.9):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    keep = sorted_p.cumsum(-1) - sorted_p < top_p
    sorted_p = sorted_p * keep
    sorted_p = sorted_p / sorted_p.sum()
    return idx[torch.multinomial(sorted_p, 1)]`,
    caseTitle: "Nucleus sampling 論文",
    case: "「The Curious Case of Neural Text Degeneration」論文分析了以最大機率為目標的解碼容易產生退化文字，並提出 nucleus（top-p）取樣。Hugging Face 文件整理了 Transformers 內可用的多種生成策略。",
    related: ["speculative-decoding", "transformer", "prompt-engineering", "latency-throughput"],
    sources: [
      ["The Curious Case of Neural Text Degeneration", "https://arxiv.org/abs/1904.09751"],
      ["Hugging Face text generation strategies", "https://huggingface.co/docs/transformers/generation_strategies"],
    ],
  },
  {
    id: "vllm",
    title: "vLLM",
    zh: "vLLM 推論引擎",
    category: "Inference Optimization",
    level: "Core focus",
    status: "published",
    summary:
      "開源的 LLM 推論與服務引擎，核心技術包含 PagedAttention 與 continuous batching，並提供 OpenAI 相容的 API 伺服器。",
    intuition:
      "像一個專為大量併發的 LLM 請求設計的調度中心：負責把 GPU 記憶體切成小格、動態安排誰先算，讓顯卡盡量不閒著。",
    mechanism: [
      "請求進入排程器，排程器以 iteration 為單位決定哪些請求進入本次的 prefill 或 decode。",
      "KV Cache 以固定大小的 block 管理（PagedAttention），減少記憶體碎片並支援前綴共享等進階功能。",
      "模型執行端支援張量平行、量化、多種 attention 後端與推測式解碼等選項，具體支援範圍隨版本更新。",
      "對外可啟動 OpenAI 相容的伺服器，方便以相同的客戶端程式碼對接不同模型。",
    ],
    diagram: [
      "HTTP / Python API 請求",
      "Scheduler（continuous batching）",
      "KV block 管理（PagedAttention）",
      "Model executor（GPU）",
      "串流回傳 token",
    ],
    code: `# 啟動 OpenAI 相容伺服器（參數以官方文件為準）
# vllm serve your-model --port 8000

from vllm import LLM, SamplingParams
llm = LLM(model="your-model")
out = llm.generate(["你好"], SamplingParams(max_tokens=64))`,
    caseTitle: "PagedAttention 論文",
    case: "vLLM 團隊在論文中提出 PagedAttention，並報告相對於當時既有系統的吞吐量提升。vLLM 是開源專案，功能與效能隨版本演進，採用前建議用自己的模型與流量實測。",
    related: ["paged-attention", "continuous-batching", "kv-cache", "model-serving"],
    sources: [
      ["vLLM documentation", "https://docs.vllm.ai/en/latest/"],
      ["PagedAttention paper", "https://arxiv.org/abs/2309.06180"],
    ],
  },
];
