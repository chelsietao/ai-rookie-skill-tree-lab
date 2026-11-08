import type { Concept, ConceptPaper } from "@/types";

type Draft = Omit<Concept, "category" | "status" | "level"> & {
  level?: Concept["level"];
};

const c = (draft: Draft): Concept => ({
  level: "Reference",
  ...draft,
  category: "Inference Optimization",
  status: "published",
});

const p = (
  title: string,
  venue: string,
  year: number,
  url: string,
  note: string
): ConceptPaper => ({ title, venue, year, url, note });

const VLLM_PAPER = (note: string) =>
  p(
    "Efficient Memory Management for Large Language Model Serving with PagedAttention",
    "SOSP",
    2023,
    "https://arxiv.org/abs/2309.06180",
    note
  );
const SARATHI_PAPER = (note: string) =>
  p(
    "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
    "OSDI",
    2024,
    "https://arxiv.org/abs/2403.02310",
    note
  );
const DISTSERVE_PAPER = (note: string) =>
  p(
    "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving",
    "OSDI",
    2024,
    "https://arxiv.org/abs/2401.09670",
    note
  );
const ORCA_PAPER = (note: string) =>
  p(
    "Orca: A Distributed Serving System for Transformer-Based Generative Models",
    "OSDI",
    2022,
    "https://www.usenix.org/conference/osdi22/presentation/yu",
    note
  );
const MOONCAKE_PAPER = (note: string) =>
  p(
    "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving",
    "arXiv",
    2024,
    "https://arxiv.org/abs/2407.00079",
    note
  );
const WHITE_PAPER = (note: string) =>
  p(
    "A White Paper on Neural Network Quantization",
    "arXiv",
    2021,
    "https://arxiv.org/abs/2106.08295",
    note
  );
const SGLANG_PAPER = (note: string) =>
  p(
    "SGLang: Efficient Execution of Structured Language Model Programs",
    "NeurIPS",
    2024,
    "https://arxiv.org/abs/2312.07104",
    note
  );

const VLLM_DOCS: [string, string] = ["vLLM documentation", "https://docs.vllm.ai/"];
const HF_QUANT: [string, string] = [
  "Hugging Face Transformers: Quantization",
  "https://huggingface.co/docs/transformers/main/en/quantization/overview",
];
const PT_QUANT: [string, string] = [
  "PyTorch: Quantization",
  "https://pytorch.org/docs/stable/quantization.html",
];

export const inferenceExtendedConcepts: Concept[] = [
  c({
    id: "time-to-first-token",
    title: "Time to First Token (TTFT)",
    zh: "首個 Token 延遲",
    level: "Core focus",
    summary: "從送出請求到收到第一個輸出 token 所花的時間，是使用者感受「有沒有反應」的主要指標。",
    intuition: "像餐廳點餐後等第一道菜上桌的時間。後面上菜再快，第一道等太久，客人還是覺得慢。",
    mechanism: [
      "TTFT 包含網路傳輸、排隊等待，以及處理整段 prompt 的 prefill 時間。",
      "prompt 越長，prefill 越久；系統繁忙時，排隊時間常成為主要來源。",
      "前綴快取可以略過已計算過的 prompt 部分，是降低 TTFT 的常見手法。",
    ],
    diagram: ["請求送達", "排隊等待", "Prefill 處理 prompt", "產生第一個 token"],
    code: `import time

start = time.perf_counter()
for i, token in enumerate(stream):  # stream 為串流回應
    if i == 0:
        ttft = time.perf_counter() - start
        print(f"TTFT: {ttft:.3f}s")`,
    caseTitle: "服務延遲指標",
    case: "DistServe 與 Sarathi-Serve 等 LLM 服務研究都把 TTFT 列為核心延遲指標，並與每個 token 的生成延遲分開衡量，因為兩者受不同階段影響。",
    related: ["time-per-output-token", "prefix-caching", "chunked-prefill", "latency-throughput"],
    sources: [["vLLM: Metrics", "https://docs.vllm.ai/en/latest/design/metrics.html"]],
    papers: [DISTSERVE_PAPER("把 TTFT 與 TPOT 分開作為 SLO，說明兩者受不同階段影響。")],
  }),
  c({
    id: "time-per-output-token",
    title: "Time per Output Token (TPOT)",
    zh: "每個輸出 Token 延遲",
    level: "Core focus",
    summary: "產生第一個 token 之後，平均每個 token 需要的時間，決定回應的串流速度。",
    intuition: "像打字速度。TTFT 是開始打字前的停頓，TPOT 是開始之後每個字打多快。",
    mechanism: [
      "TPOT 主要由 decode 階段決定，一次前向傳遞通常產生一個 token。",
      "同一批次的請求越多，單一請求的 TPOT 往往越長，但整體吞吐量較高。",
      "若 prefill 與 decode 共用 GPU，新的長 prompt 可能使進行中的請求出現明顯卡頓。",
    ],
    diagram: ["第一個 token", "Decode 迭代", "下一個 token", "重複直到結束"],
    code: `# TPOT = (總生成時間 - TTFT) / (輸出 token 數 - 1)
def tpot(total_time, ttft, num_tokens):
    return (total_time - ttft) / max(num_tokens - 1, 1)`,
    caseTitle: "為何要與 TTFT 分開看",
    case: "研究常以 TPOT 衡量互動應用的流暢度。Sarathi-Serve 指出 prefill 會讓 decode 停頓（generation stall），因此改用 chunked prefill 來維持穩定的 TPOT。",
    related: ["time-to-first-token", "continuous-batching", "chunked-prefill", "goodput"],
    sources: [["vLLM: Metrics", "https://docs.vllm.ai/en/latest/design/metrics.html"]],
    papers: [SARATHI_PAPER("分析 generation stall 對 token 間延遲的影響。")],
  }),
  c({
    id: "goodput",
    title: "Goodput",
    zh: "有效吞吐量",
    summary: "單位時間內真正符合延遲目標（SLO）的請求數，比單純的吞吐量更貼近服務品質。",
    intuition: "外送平台不只看一小時送了幾單，還要看有幾單準時送達。逾時的訂單不算數。",
    mechanism: [
      "先定義 SLO，例如 TTFT 與 TPOT 的上限。",
      "只有兩者都達標的請求才計入 goodput。",
      "把批次開得太大會提高吞吐量，但可能讓更多請求違反 SLO，使 goodput 下降。",
    ],
    diagram: ["全部請求", "檢查是否符合 SLO", "只計入達標請求", "計算 Goodput"],
    code: `def goodput(requests, duration, ttft_slo, tpot_slo):
    ok = [r for r in requests
          if r.ttft <= ttft_slo and r.tpot <= tpot_slo]
    return len(ok) / duration`,
    caseTitle: "DistServe 的最佳化目標",
    case: "DistServe 以每 GPU 的 goodput 作為主要最佳化目標，並據此決定 prefill 與 decode 分開部署的方式。",
    related: ["latency-throughput", "time-to-first-token", "time-per-output-token", "prefill-decode-disaggregation"],
    sources: [["DistServe (arXiv)", "https://arxiv.org/abs/2401.09670"]],
    papers: [DISTSERVE_PAPER("以符合 SLO 的 goodput 取代單純的吞吐量作為目標。")],
  }),
  c({
    id: "iteration-level-scheduling",
    title: "Iteration-Level Scheduling",
    zh: "迭代層級排程",
    summary: "每產生一個 token 就重新決定批次內容，讓完成的請求立刻離開、新請求立刻加入。",
    intuition: "公車每到一站都能上下客，不必等全車乘客都到終點才讓下一批人上車。",
    mechanism: [
      "傳統的請求層級批次要等最長的請求結束，先完成的請求只能空等。",
      "排程器改在每次迭代之間檢查佇列，把空出來的位置補上新請求。",
      "這需要引擎能處理批次中長度與進度各不相同的序列。",
    ],
    diagram: ["一次模型迭代", "移除已完成請求", "加入等待中請求", "下一次迭代"],
    code: `# 示意：每次迭代都更新批次
while running or waiting:
    running = [r for r in running if not r.finished]
    while waiting and has_capacity(running):
        running.append(waiting.pop(0))
    step(running)  # 每個請求前進一個 token`,
    caseTitle: "現代 LLM 服務的基礎",
    case: "Orca 提出這個做法，vLLM 與 Text Generation Inference 等服務框架也都提供類似的動態批次能力。",
    related: ["continuous-batching", "vllm", "text-generation-inference", "request-preemption"],
    sources: [["Orca (USENIX OSDI 2022)", "https://www.usenix.org/conference/osdi22/presentation/yu"]],
    papers: [ORCA_PAPER("提出 iteration-level scheduling 與 selective batching。")],
  }),
  c({
    id: "fair-llm-scheduling",
    title: "Fair LLM Scheduling",
    zh: "LLM 公平排程",
    summary: "在多個使用者共用 GPU 時，依實際消耗的運算資源分配服務，避免重度使用者擠壓其他人。",
    intuition: "共用一台影印機時，不是一人印一次就算公平。一次印一千張的人，應該被算成用得比較多。",
    mechanism: [
      "每個請求的成本取決於輸入與輸出 token 數，事先難以準確預估。",
      "VTC 以虛擬計數器累計每位使用者消耗的服務量，並優先服務累計最少的使用者。",
      "計數隨生成過程即時更新，因此不需要事先知道輸出長度。",
    ],
    diagram: ["各使用者請求佇列", "比較虛擬計數器", "挑選累計最少者", "生成時更新計數"],
    code: `# 示意：Virtual Token Counter
user = min(backlogged_users, key=lambda u: counter[u])
req = queue[user].pop(0)
counter[user] += w_in * req.input_tokens
# 之後每產生一個 token：counter[user] += w_out`,
    caseTitle: "多租戶服務",
    case: "多租戶 LLM 服務常以速率限制控管用量，但速率限制可能讓容量閒置。VTC 論文提出以公平排程在維持使用率的同時分配資源。",
    related: ["iteration-level-scheduling", "shortest-job-first-llm", "goodput", "interference-isolation"],
    sources: [["Fairness in Serving LLMs (USENIX OSDI 2024)", "https://www.usenix.org/conference/osdi24/presentation/sheng"]],
    papers: [p("Fairness in Serving Large Language Models", "OSDI", 2024, "https://arxiv.org/abs/2401.00588", "提出 Virtual Token Counter 排程演算法。")],
  }),
  c({
    id: "request-preemption",
    title: "Request Preemption",
    zh: "請求搶占",
    summary: "GPU 記憶體不足時，暫停部分進行中的請求以釋放 KV Cache，等資源空出來再恢復。",
    intuition: "停車場滿了，請部分車輛暫時開走。之後回來，可能要從頭排隊，也可能保留原本的資料繼續。",
    mechanism: [
      "輸出長度事先未知，執行中的 KV Cache 可能超過預留空間。",
      "Swap 把 KV Cache 暫存到 CPU 記憶體，恢復時再搬回 GPU。",
      "Recompute 直接丟棄快取，恢復時重新計算；對短序列可能比 swap 更省。",
    ],
    diagram: ["GPU 記憶體不足", "選擇被搶占的請求", "Swap 或 Recompute", "資源釋放後恢復"],
    code: `# 示意
if out_of_kv_blocks():
    victim = running.pop()          # 通常選最晚到的請求
    if policy == "swap":
        move_to_cpu(victim.kv_blocks)
    else:
        free(victim.kv_blocks)      # 恢復時重算
    waiting.insert(0, victim)`,
    caseTitle: "vLLM 的搶占機制",
    case: "vLLM 在 KV Cache 空間不足時會搶占請求，論文中比較了 swapping 與 recomputation 兩種恢復方式。",
    related: ["paged-attention", "vllm", "kv-cache", "copy-on-write-kv-cache"],
    sources: [VLLM_DOCS],
    papers: [VLLM_PAPER("討論搶占策略，並比較 swapping 與 recomputation。")],
  }),
  c({
    id: "kv-cache-prefetching",
    title: "KV Cache Prefetching",
    zh: "KV Cache 預先載入",
    summary: "把部分 KV Cache 放在 CPU 記憶體，只預先把下一步可能重要的部分搬回 GPU。",
    intuition: "書架放不下所有書，就先預測接下來會翻哪幾本，提前從倉庫拿出來。",
    mechanism: [
      "長上下文的 KV Cache 可能超過 GPU 記憶體容量。",
      "InfiniGen 利用模型前一層的資訊，推測下一層哪些 token 的 attention 權重較高。",
      "只載入這些重要部分，在減少傳輸量的同時盡量維持輸出品質。",
    ],
    diagram: ["完整 KV 放在 CPU", "預測重要 token", "只載入部分 KV 到 GPU", "計算 Attention"],
    code: `# 示意：只取預測為重要的位置
scores = predict_attention(prev_layer_output)
idx = scores.topk(k).indices
k_gpu = k_cpu[:, idx].to("cuda")
v_gpu = v_cpu[:, idx].to("cuda")`,
    caseTitle: "長上下文推論",
    case: "InfiniGen 針對 KV Cache 在 CPU 與 GPU 之間搬移的成本，提出選擇性預先載入的方法。",
    related: ["kv-cache", "kv-cache-offloading", "kv-cache-quantization", "cpu-gpu-offloading"],
    sources: [["InfiniGen (USENIX OSDI 2024)", "https://www.usenix.org/conference/osdi24/presentation/lee"]],
    papers: [p("InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management", "OSDI", 2024, "https://arxiv.org/abs/2406.19707", "提出依重要性動態載入 KV Cache。")],
  }),
  c({
    id: "kv-cache-quantization",
    title: "KV Cache Quantization",
    zh: "KV Cache 量化",
    summary: "以較少位元儲存 KV Cache 以節省記憶體，代價是多一點編解碼時間與可能的精度損失。",
    intuition: "把高解析度照片壓成較小的檔案，手機就放得下更多張，只是打開時要解壓，畫質也可能略降。",
    mechanism: [
      "KV Cache 隨序列長度成長，通常是推論時記憶體的主要消耗之一。",
      "KIVI 的觀察是 Key 與 Value 對量化的敏感度不同，因此採取不對稱的量化方式。",
      "Key 以通道為單位、Value 以 token 為單位量化，近期 token 則保留全精度。",
    ],
    diagram: ["新 token 的 K/V", "暫存全精度", "超過視窗後量化", "計算時解量化"],
    code: `# Hugging Face Transformers 的量化快取
out = model.generate(
    **inputs,
    max_new_tokens=128,
    cache_implementation="quantized",
    cache_config={"backend": "quanto", "nbits": 4},
)`,
    caseTitle: "Hugging Face 的量化快取",
    case: "Hugging Face Transformers 提供 QuantizedCache，讓使用者以少量額外運算換取較低的 KV Cache 記憶體用量。",
    related: ["kv-cache", "quantization", "kv-cache-prefetching", "grouped-query-attention"],
    sources: [["Hugging Face: Caching", "https://huggingface.co/docs/transformers/main/en/cache_explanation"]],
    papers: [p("KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", "ICML", 2024, "https://arxiv.org/abs/2402.02750", "提出 Key 與 Value 不對稱的快取量化方式。")],
  }),
  c({
    id: "kv-cache-offloading",
    title: "KV Cache Offloading",
    zh: "KV Cache 卸載",
    summary: "把 KV Cache 存到 CPU 記憶體或 SSD，以便重用先前算過的結果，減少重複的 prefill。",
    intuition: "把算好的筆記存進檔案櫃，下次遇到相同題目就拿出來用，不必重新計算。",
    mechanism: [
      "GPU 記憶體有限，無法長期保留所有請求的 KV Cache。",
      "把快取卸載到容量較大的儲存層，等於用儲存空間換取運算量。",
      "是否划算取決於命中率，以及搬回快取和重新計算之間的時間差。",
    ],
    diagram: ["GPU 記憶體", "CPU DRAM", "SSD / 遠端儲存", "命中時搬回 GPU"],
    code: `# 示意
key = hash(prompt_prefix)
if key in kv_store:
    past = kv_store.get(key).to("cuda")
    logits = model(new_tokens, past_key_values=past)
else:
    logits = model(prompt_tokens)`,
    caseTitle: "Mooncake 的架構",
    case: "Mooncake 是 Moonshot AI 為 Kimi 服務設計的架構，以 KV Cache 為核心，把 CPU、DRAM 與 SSD 組成分散式快取池。",
    related: ["kv-cache", "prefix-caching", "rdma-kv-transfer", "cache-aware-routing"],
    sources: [["Mooncake (GitHub)", "https://github.com/kvcache-ai/Mooncake"]],
    papers: [MOONCAKE_PAPER("以 KV Cache 為中心，說明分層快取與排程設計。")],
  }),
  c({
    id: "cache-aware-routing",
    title: "Cache-Aware Routing",
    zh: "快取感知路由",
    summary: "把請求送到已經有相同 prompt 前綴快取的副本，同時兼顧各副本的負載。",
    intuition: "要找資料時，先去已經把資料攤在桌上的同事那裡，但如果他手上工作太滿，還是得找別人。",
    mechanism: [
      "前綴快取存在各副本的 GPU 上，一般的輪詢式負載平衡無法善用。",
      "路由器追蹤各副本持有哪些前綴，估算命中的好處與排隊的代價。",
      "只看快取會讓熱門前綴集中在少數副本，因此需要與負載一起考量。",
    ],
    diagram: ["新請求", "比對各副本的前綴", "估算命中與負載", "送往最佳副本"],
    code: `# 示意
def route(prompt, replicas):
    def cost(r):
        hit = r.longest_prefix_match(prompt)
        return r.queue_time() + r.prefill_time(len(prompt) - hit)
    return min(replicas, key=cost)`,
    caseTitle: "分散式前綴排程",
    case: "Preble 研究在多 GPU 環境中同時考慮提示詞重用與負載平衡；SGLang 也提供具快取感知能力的路由元件。",
    related: ["prefix-caching", "radix-attention", "kv-cache-offloading", "live-migration-llm"],
    sources: [["SGLang documentation", "https://docs.sglang.ai/"]],
    papers: [p("Preble: Efficient Distributed Prompt Scheduling for LLM Serving", "ICLR", 2025, "https://arxiv.org/abs/2407.00023", "同時考量提示詞快取與負載的分散式排程。")],
  }),
  c({
    id: "radix-attention",
    title: "RadixAttention",
    zh: "基數樹注意力快取",
    summary: "以基數樹管理 KV Cache，讓擁有共同前綴的多個請求自動共用已計算的結果。",
    intuition: "把句子存成字典樹，開頭相同的句子共用同一段路徑，不必各存一份。",
    mechanism: [
      "基數樹的邊對應 token 序列，節點則對應已存在的 KV Cache。",
      "新請求沿著樹走到最長匹配，只需計算其餘部分。",
      "記憶體不足時以 LRU 淘汰葉節點，並與排程策略配合以提高命中率。",
    ],
    diagram: ["請求 token 序列", "在基數樹找最長匹配", "重用對應 KV", "只計算新增部分"],
    code: `# 示意
node, matched = radix_tree.longest_prefix(tokens)
past = node.kv_cache
new_tokens = tokens[matched:]
logits, new_kv = model(new_tokens, past_key_values=past)
radix_tree.insert(tokens, new_kv)`,
    caseTitle: "SGLang 的快取機制",
    case: "SGLang 提出 RadixAttention，用於多輪對話與結構化程式中常見的共同前綴場景。",
    related: ["prefix-caching", "sglang", "kv-cache", "cache-aware-routing"],
    sources: [["SGLang documentation", "https://docs.sglang.ai/"]],
    papers: [SGLANG_PAPER("提出 RadixAttention 以自動重用共同前綴。")],
  }),
  c({
    id: "structured-output",
    title: "Structured Output Decoding",
    zh: "結構化輸出解碼",
    summary: "在生成時限制可選的 token，保證輸出符合 JSON Schema、正規表示式或文法。",
    intuition: "填表格時，日期欄只讓你選日期。模型每產生一個 token，都只能從目前合法的選項裡挑。",
    mechanism: [
      "把 schema 或文法轉成有限狀態機或下推自動機。",
      "每一步依目前狀態算出合法 token，並把其餘 token 的 logits 遮蔽。",
      "預先計算遮罩並快取可以降低額外成本，否則每步檢查詞彙表會拖慢生成。",
    ],
    diagram: ["JSON Schema / 文法", "編譯成自動機", "遮蔽不合法 token", "取樣下一個 token", "更新狀態"],
    code: `# 概念示意
mask = automaton.allowed_tokens(state)
logits[~mask] = float("-inf")
token = sample(logits)
state = automaton.advance(state, token)`,
    caseTitle: "推論引擎內建支援",
    case: "Outlines 以有限狀態機實作引導式生成；XGrammar 專注於高效的文法約束，並已被多個推論引擎整合。",
    related: ["sampling-decoding", "sglang", "vllm", "function-calling"],
    sources: [["Outlines (GitHub)", "https://github.com/dottxt-ai/outlines"], ["XGrammar (GitHub)", "https://github.com/mlc-ai/xgrammar"]],
    papers: [
      p("Efficient Guided Generation for Large Language Models", "arXiv", 2023, "https://arxiv.org/abs/2307.09702", "以有限狀態機加速正規表示式與文法約束。"),
      p("XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", "arXiv", 2024, "https://arxiv.org/abs/2411.15100", "透過預先計算遮罩提高文法約束的效率。"),
    ],
  }),
  c({
    id: "sglang",
    title: "SGLang",
    zh: "SGLang 推論框架",
    summary: "開源的 LLM 推論引擎與程式框架，提供 RadixAttention、結構化輸出與連續批次等能力。",
    intuition: "同時提供高速廚房與點餐語言：既負責快速出餐，也讓人描述複雜的多步驟流程。",
    mechanism: [
      "前端語言讓使用者描述多次生成、控制流程與平行呼叫。",
      "後端引擎負責批次、快取管理與 GPU 執行。",
      "前後端協作，讓共同前綴與結構化輸出可以被有效利用。",
    ],
    diagram: ["SGLang 程式 / OpenAI 相容 API", "Runtime 排程", "RadixAttention", "GPU 執行"],
    code: `# 啟動與 OpenAI 相容的服務
# python -m sglang.launch_server --model-path <model> --port 30000
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
resp = client.chat.completions.create(
    model="<model>", messages=[{"role": "user", "content": "你好"}])`,
    caseTitle: "開源推論引擎",
    case: "SGLang 由學界發起並持續開源維護，提供與 OpenAI 相容的介面，常與 vLLM 一同被比較。",
    related: ["radix-attention", "vllm", "structured-output", "prefix-caching"],
    sources: [["SGLang documentation", "https://docs.sglang.ai/"], ["SGLang (GitHub)", "https://github.com/sgl-project/sglang"]],
    papers: [SGLANG_PAPER("SGLang 的原始論文。")],
  }),
  c({
    id: "llama-cpp",
    title: "llama.cpp",
    zh: "llama.cpp 推論引擎",
    summary: "以 C/C++ 實作的 LLM 推論引擎，著重在 CPU 與消費級硬體上執行量化模型。",
    intuition: "把大型工廠的生產線縮小成家用機器，雖然規模較小，但在一般電腦上就能運作。",
    mechanism: [
      "建立在 ggml 張量函式庫上，依各平台提供手寫的 SIMD 與 GPU 後端。",
      "使用 GGUF 格式儲存量化權重與中繼資料。",
      "可把部分層放到 GPU，其餘留在 CPU，以因應記憶體限制。",
    ],
    diagram: ["GGUF 模型檔", "mmap 載入權重", "ggml 運算圖", "CPU / GPU 後端", "輸出 token"],
    code: `# 以量化模型啟動，並將 32 層放到 GPU
llama-cli -m model-Q4_K_M.gguf -ngl 32 -p "解釋 KV Cache" -n 128`,
    caseTitle: "本機推論生態",
    case: "許多本機 LLM 工具都建立在 llama.cpp 或其 GGUF 格式之上，例如 Ollama 就使用它作為推論基礎之一。",
    related: ["gguf", "quantization", "cpu-gpu-offloading", "kv-cache"],
    sources: [["llama.cpp (GitHub)", "https://github.com/ggml-org/llama.cpp"]],
  }),
  c({
    id: "medusa-decoding",
    title: "Medusa",
    zh: "Medusa 多頭解碼",
    summary: "在模型上增加多個解碼頭，一次預測多個未來 token，不需另一個草稿模型。",
    intuition: "不必另外請一位助理打草稿，而是讓自己多長幾隻手，同時寫下接下來幾個字。",
    mechanism: [
      "每個 Medusa head 負責預測未來第 k 個位置的 token。",
      "組合各頭的候選，形成樹狀的候選序列，並以 tree attention 一次驗證。",
      "比起 speculative decoding，不必處理第二個模型的記憶體與對齊問題。",
    ],
    diagram: ["基礎模型特徵", "多個 Medusa heads", "候選序列樹", "Tree Attention 驗證", "接受最長前綴"],
    code: `# 示意
hidden = base_model(tokens)
candidates = [head(hidden) for head in medusa_heads]
tree = build_candidate_tree(candidates)
logits = base_model(tree, attention_mask=tree.mask)
accepted = accept_longest_valid_prefix(tree, logits)`,
    caseTitle: "無草稿模型的加速",
    case: "Medusa 論文提出以額外的解碼頭加速生成，可以只訓練新增的頭，而不更動基礎模型的權重。",
    related: ["speculative-decoding", "eagle-decoding", "lookahead-decoding", "prompt-lookup-decoding"],
    sources: [["Medusa (GitHub)", "https://github.com/FasterDecoding/Medusa"]],
    papers: [p("Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", "ICML", 2024, "https://arxiv.org/abs/2401.10774", "Medusa 的原始論文。")],
  }),
  c({
    id: "eagle-decoding",
    title: "EAGLE",
    zh: "EAGLE 特徵層推測解碼",
    summary: "在特徵層做外推，以輕量模型預測下一步的隱藏狀態，再透過主模型驗證。",
    intuition: "不直接猜下一個字，而是先猜主模型腦中的想法，再由這個想法推出字，通常比較準。",
    mechanism: [
      "草稿模型同時接收已生成的 token 與目前特徵，預測下一步的特徵。",
      "利用目標模型的輸出頭把特徵轉成 token 分布。",
      "再依 speculative sampling 的規則接受或拒絕，以維持輸出分布。",
    ],
    diagram: ["目標模型特徵", "輕量自迴歸草稿", "預測未來特徵", "產生候選 token", "目標模型驗證"],
    code: `# 示意
feat = target.hidden_state(tokens)
for _ in range(k):
    feat = eagle(feat, last_token)
    last_token = sample(target.lm_head(feat))
    drafts.append(last_token)
verify_with_target(drafts)`,
    caseTitle: "推論引擎的支援",
    case: "vLLM 文件將 EAGLE 列為可用的推測解碼方式之一。",
    related: ["speculative-decoding", "medusa-decoding", "lookahead-decoding", "vllm"],
    sources: [["EAGLE (GitHub)", "https://github.com/SafeAILab/EAGLE"]],
    papers: [p("EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", "ICML", 2024, "https://arxiv.org/abs/2401.15077", "提出在特徵層進行推測。")],
  }),
  c({
    id: "lookahead-decoding",
    title: "Lookahead Decoding",
    zh: "前瞻解碼",
    summary: "把雅可比迭代用於生成，平行猜測未來 token，不需要額外的模型或解碼頭。",
    intuition: "不是一個字一個字寫，而是先寫一整句草稿，再同時訂正每個字，訂到穩定為止。",
    mechanism: [
      "把順序生成視為一組非線性方程，以雅可比迭代平行求解。",
      "把先前迭代的結果存成 n-gram 池，並用來驗證與擴展候選。",
      "接受率取決於模型與內容，因此實際速度提升會因情境而異。",
    ],
    diagram: ["目前序列", "平行猜測未來 token", "收集 n-gram 池", "驗證並接受"],
    code: `# 示意
guess = init_guess(tokens, window)
for _ in range(iterations):
    guess = argmax(model(tokens + guess))   # 平行更新
    ngram_pool.update(guess)
accepted = verify(tokens, ngram_pool)`,
    caseTitle: "不需草稿模型",
    case: "論文指出，這個方法不需要草稿模型，因此可省去訓練與維護額外模型的成本。",
    related: ["speculative-decoding", "medusa-decoding", "prompt-lookup-decoding", "sampling-decoding"],
    sources: [["Lookahead Decoding (LMSYS blog)", "https://lmsys.org/blog/2023-11-21-lookahead-decoding/"]],
    papers: [p("Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", "ICML", 2024, "https://arxiv.org/abs/2402.02057", "前瞻解碼的原始論文。")],
  }),
  c({
    id: "prompt-lookup-decoding",
    title: "Prompt Lookup Decoding",
    zh: "提示詞查找解碼",
    summary: "直接從 prompt 中找出與目前結尾相符的 n-gram，把接下來的內容當作草稿。",
    intuition: "改文章時常常照抄原文。既然要抄，就先把原文接下來幾個字貼上，再請模型確認。",
    mechanism: [
      "以最近幾個生成的 token，在 prompt 裡尋找相同的 n-gram。",
      "把匹配處之後的 token 視為候選，一次交由模型驗證。",
      "對摘要、程式碼編輯等輸出大量重複輸入的場景較有效。",
    ],
    diagram: ["最近 n 個 token", "在 prompt 中搜尋", "取後續 token 為草稿", "模型驗證"],
    code: `# Hugging Face Transformers
out = model.generate(
    **inputs,
    prompt_lookup_num_tokens=10,
    max_new_tokens=200,
)`,
    caseTitle: "框架內建功能",
    case: "Hugging Face Transformers 以 prompt_lookup_num_tokens 參數支援這個做法；vLLM 也提供類似的 n-gram 推測。",
    related: ["speculative-decoding", "lookahead-decoding", "medusa-decoding", "vllm"],
    sources: [["Transformers: Generation strategies", "https://huggingface.co/docs/transformers/main/en/generation_strategies"]],
    papers: [p("Fast Inference from Transformers via Speculative Decoding", "ICML", 2023, "https://arxiv.org/abs/2211.17192", "提示詞查找是其驗證框架的一種應用，只是草稿來源改成 prompt。")],
  }),
  c({
    id: "gptq",
    title: "GPTQ",
    zh: "GPTQ 權重量化",
    level: "Core focus",
    summary: "訓練後的權重量化方法，利用二階資訊逐層調整權重，以降低量化造成的誤差。",
    intuition: "把照片壓縮成較少顏色時，不只是挑最接近的顏色，還會調整相鄰像素，讓整體看起來仍然正確。",
    mechanism: [
      "逐層處理，並使用校準資料估算 Hessian。",
      "依序量化每個權重，再更新尚未量化的權重以補償誤差。",
      "常見於 4-bit 權重量化，只需少量校準資料。",
    ],
    diagram: ["原始權重", "校準資料估算 Hessian", "逐一量化權重", "補償剩餘權重", "量化模型"],
    code: `from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_id)
cfg = GPTQConfig(bits=4, dataset="c4", tokenizer=tok)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=cfg, device_map="auto")`,
    caseTitle: "開源模型的量化版本",
    case: "社群常以 GPTQ 發布 LLM 的 4-bit 版本，Hugging Face Transformers 也內建支援。",
    related: ["quantization", "awq", "smoothquant", "post-training-quantization"],
    sources: [HF_QUANT],
    papers: [p("GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", "ICLR", 2023, "https://arxiv.org/abs/2210.17323", "GPTQ 的原始論文。")],
  }),
  c({
    id: "awq",
    title: "AWQ",
    zh: "激活感知權重量化",
    summary: "依激活值的大小辨識重要的權重通道，先調整尺度再量化，以保留對輸出影響較大的資訊。",
    intuition: "壓縮音樂時，先確保人耳最敏感的頻段被保留，其他頻段再壓得多一點。",
    mechanism: [
      "用校準資料觀察激活值，判斷哪些通道比較重要。",
      "對重要通道乘上尺度因子，並在對應的輸入端除回去，讓數學上等價。",
      "再進行權重量化，降低重要通道的誤差。",
    ],
    diagram: ["校準資料", "統計激活值", "搜尋通道尺度", "權重量化", "量化模型"],
    code: `from transformers import AutoModelForCausalLM

# 載入已經用 AWQ 量化的模型
model = AutoModelForCausalLM.from_pretrained(
    "<awq-quantized-model>", device_map="auto")`,
    caseTitle: "推論框架支援",
    case: "vLLM 與 Hugging Face Transformers 都可載入 AWQ 量化模型，社群也常以此發布量化版本。",
    related: ["quantization", "gptq", "smoothquant", "vllm"],
    sources: [HF_QUANT],
    papers: [p("AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", "MLSys", 2024, "https://arxiv.org/abs/2306.00978", "AWQ 的原始論文。")],
  }),
  c({
    id: "smoothquant",
    title: "SmoothQuant",
    zh: "SmoothQuant 平滑量化",
    summary: "把激活值的量化困難轉移到權重上，讓 Transformer 也能使用 INT8 的權重與激活量化。",
    intuition: "兩個人抬東西，一人負擔太重就容易出錯。把重量分一些給另一人，兩人都比較穩。",
    mechanism: [
      "LLM 的激活值常有離群值，直接量化誤差很大。",
      "以數學上等價的尺度轉換，壓低激活值的範圍，同時放大權重。",
      "兩者都變得容易量化，再用 INT8 的方式處理。",
    ],
    diagram: ["統計激活值範圍", "計算平滑因子", "轉移尺度到權重", "INT8 量化"],
    code: `# 概念示意：對第 i 個通道
s = activation_max[i] ** alpha / weight_max[i] ** (1 - alpha)
activation[:, i] = activation[:, i] / s
weight[:, i]     = weight[:, i] * s`,
    caseTitle: "INT8 推論",
    case: "論文以 INT8 量化在 CPU 等硬體上評估 LLM 推論。Intel Neural Compressor 等工具也提供相應的實作。",
    related: ["quantization", "gptq", "awq", "llm-int8"],
    sources: [["SmoothQuant (GitHub)", "https://github.com/mit-han-lab/smoothquant"]],
    papers: [p("SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", "ICML", 2023, "https://arxiv.org/abs/2211.10438", "SmoothQuant 的原始論文。")],
  }),
  c({
    id: "quantization-aware-training",
    title: "Quantization-Aware Training (QAT)",
    zh: "量化感知訓練",
    summary: "訓練時模擬量化造成的誤差，讓模型學會適應低精度，通常比訓練後才量化更準確。",
    intuition: "知道比賽要穿重鞋，就穿著重鞋練習，正式上場時才不會不適應。",
    mechanism: [
      "在權重與激活上插入 fake quantization，前向時模擬量化。",
      "反向傳播以 straight-through estimator 等方式處理不可微分的取整。",
      "訓練後再轉換成真正的整數模型。",
    ],
    diagram: ["浮點模型", "插入 Fake Quant", "繼續訓練", "轉換成整數模型"],
    code: `import torch.ao.quantization as tq

model.train()
model.qconfig = tq.get_default_qat_qconfig("x86")
tq.prepare_qat(model, inplace=True)
train(model)
model.eval()
quantized = tq.convert(model)`,
    caseTitle: "行動裝置部署",
    case: "Google 在行動裝置的整數推論研究中提出這類訓練流程，後續被整合進 TensorFlow Lite 與 PyTorch 的工具中。",
    related: ["quantization", "post-training-quantization", "half-precision-inference", "knowledge-distillation"],
    sources: [PT_QUANT],
    papers: [p("Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", "CVPR", 2018, "https://arxiv.org/abs/1712.05877", "整數推論與量化訓練的經典論文。")],
  }),
  c({
    id: "post-training-quantization",
    title: "Post-Training Quantization (PTQ)",
    zh: "訓練後量化",
    summary: "模型訓練完成後才進行量化，不需重新訓練，是最常見也最省事的做法。",
    intuition: "衣服做好才修改尺寸，不用重新裁布，但太大幅度的修改可能不太合身。",
    mechanism: [
      "動態量化只先量化權重，激活值在執行時才計算範圍。",
      "靜態量化以校準資料統計激活值範圍，再固定 scale 與 zero-point。",
      "位元數越低、模型越敏感，精度損失就越可能明顯，需要評估。",
    ],
    diagram: ["訓練完成的模型", "校準資料統計範圍", "計算 scale / zero-point", "整數模型"],
    code: `import torch

quantized = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)`,
    caseTitle: "工具鏈支援",
    case: "PyTorch、ONNX Runtime 與 TensorRT 都提供訓練後量化工具，常作為優先嘗試的量化方式。",
    related: ["quantization", "quantization-aware-training", "gptq", "onnx-runtime"],
    sources: [PT_QUANT, ["ONNX Runtime: Quantization", "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html"]],
    papers: [WHITE_PAPER("說明 PTQ 與 QAT 的流程，以及如何除錯。")],
  }),
  c({
    id: "gguf",
    title: "GGUF",
    zh: "GGUF 模型格式",
    summary: "llama.cpp 使用的單一檔案模型格式，內含權重、tokenizer 與中繼資料，並支援多種量化。",
    intuition: "把食材、食譜和說明都裝進同一個便當盒，帶著走就能直接開吃。",
    mechanism: [
      "檔案以鍵值中繼資料開頭，接著是張量資訊與資料區塊。",
      "張量資料依對齊方式排列，可以用 mmap 直接映射到記憶體。",
      "檔名中的 Q4_K_M 等標記表示量化方式。",
    ],
    diagram: ["Header", "Metadata 鍵值", "張量資訊", "對齊後的權重資料"],
    code: `# pip install gguf
from gguf import GGUFReader

reader = GGUFReader("model-Q4_K_M.gguf")
print(list(reader.fields.keys())[:5])
print(len(reader.tensors))`,
    caseTitle: "本機模型的發布格式",
    case: "Hugging Face 上許多量化的 LLM 都以 GGUF 發布，方便 llama.cpp 與相關工具直接載入。",
    related: ["llama-cpp", "quantization", "safetensors", "cpu-gpu-offloading"],
    sources: [["ggml: GGUF specification", "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md"], ["Hugging Face: GGUF", "https://huggingface.co/docs/hub/gguf"]],
  }),
  c({
    id: "llm-int8",
    title: "LLM.int8() / bitsandbytes",
    zh: "8-bit 矩陣相乘",
    summary: "在 Transformer 大型模型上以 8-bit 矩陣相乘推論，把大多數權重壓縮，同時將離群值維持高精度。",
    intuition: "班上大多數人用簡化成績單，少數特別的學生仍保留詳細紀錄，整體就省空間又不失真。",
    mechanism: [
      "矩陣中的離群值對乘法結果影響大，因此先將它們分離。",
      "一般部分以 INT8 矩陣相乘，離群值則以 FP16 計算後加回。",
      "推論時可以顯著節省記憶體，但會增加一些運算開銷。",
    ],
    diagram: ["權重矩陣", "分離離群值", "INT8 相乘", "FP16 相乘離群值", "相加結果"],
    code: `from transformers import AutoModelForCausalLM, BitsAndBytesConfig

cfg = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=cfg, device_map="auto")`,
    caseTitle: "bitsandbytes 函式庫",
    case: "bitsandbytes 實作了這個方法，並與 Hugging Face Transformers 整合，常用於在有限 GPU 記憶體上載入大模型。",
    related: ["quantization", "qlora", "smoothquant", "half-precision-inference"],
    sources: [["bitsandbytes documentation", "https://huggingface.co/docs/bitsandbytes/main/en/index"]],
    papers: [p("LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", "NeurIPS", 2022, "https://arxiv.org/abs/2208.07339", "說明以離群值分解達成大型模型的 8-bit 推論。")],
  }),
  c({
    id: "weight-pruning",
    title: "Weight Pruning",
    zh: "權重剪枝",
    summary: "把影響不大的權重設為零，讓模型變小；若要加速，還需要硬體或軟體能利用稀疏性。",
    intuition: "整理樹木時，把不影響結果的小枝剪掉，樹會變輕，但主幹與形狀大致不變。",
    mechanism: [
      "通常以權重大小作為重要性的依據，把最小的權重設為零。",
      "剪枝後常需要微調模型，恢復部分準確率。",
      "非結構化剪枝的零散分布不一定能在 GPU 上帶來加速。",
    ],
    diagram: ["訓練好的模型", "依重要性排序", "移除權重", "微調", "稀疏模型"],
    code: `import torch.nn.utils.prune as prune

prune.l1_unstructured(model.fc, name="weight", amount=0.5)
prune.remove(model.fc, "weight")   # 讓剪枝結果永久生效`,
    caseTitle: "經典壓縮流程",
    case: "Deep Compression 以剪枝、權重共享與 Huffman 編碼來壓縮模型；PyTorch 也提供 torch.nn.utils.prune 工具。",
    related: ["structured-pruning", "quantization", "knowledge-distillation", "hardware-aware-nas"],
    sources: [["PyTorch: Pruning tutorial", "https://pytorch.org/tutorials/intermediate/pruning_tutorial.html"]],
    papers: [
      p("Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", "ICLR", 2016, "https://arxiv.org/abs/1510.00149", "結合剪枝與量化的經典壓縮流程。"),
      p("The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", "ICLR", 2019, "https://arxiv.org/abs/1803.03635", "探討稀疏子網路為何能被訓練。"),
    ],
  }),
  c({
    id: "structured-pruning",
    title: "Structured Pruning",
    zh: "結構化剪枝",
    summary: "整個移除通道、卷積核或注意力頭，得到較小的密集模型，不需特殊硬體就能加速。",
    intuition: "不是在每個房間拆幾塊磚，而是直接拆掉整個房間。房子變小，走起來也真的比較快。",
    mechanism: [
      "以範數等指標評估整個結構單元的重要性。",
      "移除單元時必須同步調整上下游的張量形狀。",
      "壓縮率通常低於非結構化剪枝，但比較容易真正提升速度。",
    ],
    diagram: ["評估通道重要性", "移除通道", "調整相鄰層形狀", "微調模型"],
    code: `import torch.nn.utils.prune as prune

# 依 L2 範數，沿 dim=0 移除 30% 的通道
prune.ln_structured(model.conv, name="weight", amount=0.3, n=2, dim=0)`,
    caseTitle: "卷積網路的通道剪枝",
    case: "Li 等人的研究以 L1 範數篩選 CNN 的卷積核，移除後的模型仍是一般的密集網路，因此可直接以既有框架執行。",
    related: ["weight-pruning", "quantization", "hardware-aware-nas", "knowledge-distillation"],
    sources: [["PyTorch: Pruning tutorial", "https://pytorch.org/tutorials/intermediate/pruning_tutorial.html"]],
    papers: [p("Pruning Filters for Efficient ConvNets", "ICLR", 2017, "https://arxiv.org/abs/1608.08710", "以 L1 範數移除卷積核。")],
  }),
  c({
    id: "model-routing",
    title: "Model Routing",
    zh: "模型路由",
    summary: "依問題難度把請求分流到不同模型，簡單的給小模型，困難的給大模型，以降低成本。",
    intuition: "客服先讓第一線處理常見問題，較難的才轉給資深專員。",
    mechanism: [
      "路由器預測小模型是否足以回答這個問題。",
      "以閾值調整成本與品質之間的取捨。",
      "路由的判斷本身也有成本與錯誤率，需要實際評估。",
    ],
    diagram: ["使用者請求", "路由器預測難度", "小模型或大模型", "回傳結果"],
    code: `# 示意
score = router.predict(prompt)   # 預測需要大模型的機率
model = strong_model if score > threshold else weak_model
answer = model.generate(prompt)`,
    caseTitle: "RouteLLM",
    case: "RouteLLM 提出以偏好資料訓練路由器，並開源相關框架，研究如何在維持品質的同時減少大模型的呼叫。",
    related: ["semantic-caching", "knowledge-distillation", "model-serving", "latency-throughput"],
    sources: [["RouteLLM (GitHub)", "https://github.com/lm-sys/RouteLLM"]],
    papers: [p("RouteLLM: Learning to Route LLMs with Preference Data", "ICLR", 2025, "https://arxiv.org/abs/2406.18665", "以偏好資料訓練模型路由器。")],
  }),
  c({
    id: "attention-sinks",
    title: "Attention Sinks",
    zh: "注意力匯聚點",
    summary: "模型傾向把大量注意力放在開頭幾個 token 上，保留它們就能在串流生成時穩定地捨棄舊內容。",
    intuition: "長時間會議只留最近的筆記和開頭的議程。少了議程，大家反而容易迷失方向。",
    mechanism: [
      "Softmax 要求注意力權重總和為 1，模型常把多餘的注意力放到開頭 token。",
      "若滑動視窗把開頭 token 丟掉，輸出品質可能迅速下降。",
      "StreamingLLM 保留前幾個 token 與最近的視窗，其餘 KV Cache 則捨棄。",
    ],
    diagram: ["開頭 sink tokens", "捨棄中間 KV", "保留最近視窗", "持續生成"],
    code: `# 示意：固定保留開頭與最近的 KV
keep = list(range(n_sink)) + list(range(seq_len - window, seq_len))
k_cache = k_cache[:, :, keep]
v_cache = v_cache[:, :, keep]`,
    caseTitle: "長對話的串流生成",
    case: "StreamingLLM 指出可以用固定大小的 KV Cache 持續生成很長的文字，同時說明注意力匯聚點在此扮演的角色。",
    related: ["kv-cache", "context-window", "kv-cache-offloading", "grouped-query-attention"],
    sources: [["streaming-llm (GitHub)", "https://github.com/mit-han-lab/streaming-llm"]],
    papers: [p("Efficient Streaming Language Models with Attention Sinks", "ICLR", 2024, "https://arxiv.org/abs/2309.17453", "提出 attention sink 現象與 StreamingLLM。")],
  }),
  c({
    id: "prompt-compression",
    title: "Prompt Compression",
    zh: "提示詞壓縮",
    summary: "在送入模型之前刪除資訊量較低的 token，以降低 prefill 成本與上下文佔用。",
    intuition: "發電報時去掉不影響理解的字，既省錢又快，但刪太多就可能看不懂。",
    mechanism: [
      "使用小模型估算每個 token 的資訊量或困惑度。",
      "依壓縮率保留重要 token，其餘刪除。",
      "可能遺失關鍵細節，所以要針對任務評估品質。",
    ],
    diagram: ["原始 prompt", "小模型評估", "刪除低資訊 token", "送入主模型"],
    code: `from llmlingua import PromptCompressor

compressor = PromptCompressor()
result = compressor.compress_prompt(
    long_prompt, rate=0.5, force_tokens=["\\n", "?"])
print(result["compressed_prompt"])`,
    caseTitle: "LLMLingua",
    case: "Microsoft 的 LLMLingua 以小型語言模型壓縮 prompt，並開源了相關工具。",
    related: ["context-window", "time-to-first-token", "chunked-prefill", "rag"],
    sources: [["LLMLingua (GitHub)", "https://github.com/microsoft/LLMLingua"]],
    papers: [p("LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", "EMNLP", 2023, "https://arxiv.org/abs/2310.05736", "以小模型估算 token 資訊量來壓縮 prompt。")],
  }),
  c({
    id: "scaled-dot-product-attention",
    title: "PyTorch SDPA",
    zh: "PyTorch 縮放點積注意力",
    summary: "PyTorch 內建的注意力函式，會依硬體與輸入選擇 FlashAttention 等最佳實作。",
    intuition: "你只要說要過河，系統會依現場狀況挑橋、渡輪或隧道，不用自己比較每一種。",
    mechanism: [
      "統一使用 torch.nn.functional.scaled_dot_product_attention 介面。",
      "後端可依裝置與輸入選擇 FlashAttention、記憶體高效注意力或數學定義版本。",
      "搭配 attention mask 或 need_weights 等條件時，可用的後端可能不同。",
    ],
    diagram: ["Q, K, V", "SDPA 介面", "選擇最佳實作", "注意力輸出"],
    code: `import torch.nn.functional as F

# q, k, v: (batch, heads, seq, head_dim)
out = F.scaled_dot_product_attention(
    q, k, v, is_causal=True)`,
    caseTitle: "框架的預設選擇",
    case: "Hugging Face Transformers 在支援的模型與環境中，可透過 attn_implementation=\"sdpa\" 使用這個介面。",
    related: ["flash-attention", "attention", "gpu-memory", "kv-cache"],
    sources: [["PyTorch: scaled_dot_product_attention", "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"]],
    papers: [p("FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "ICLR", 2024, "https://arxiv.org/abs/2307.08691", "SDPA 在 GPU 上可使用的 FlashAttention 後端。")],
  }),
  c({
    id: "adaptive-batching",
    title: "Adaptive Batching",
    zh: "自適應批次",
    summary: "依目前負載動態調整批次大小與等待時間，兼顧延遲與吞吐量。",
    intuition: "接駁車不是固定等滿才開，而是視人潮調整：人多就多載幾個，人少就早點出發。",
    mechanism: [
      "批次越大，吞吐量通常越高，但個別請求等待時間越長。",
      "系統可設定最大等待時間與最大批次，並依負載調整。",
      "Clipper 以類似 AIMD 的方式找到符合延遲目標的最大批次。",
    ],
    diagram: ["請求抵達", "等待直到批次滿或逾時", "批次推論", "依延遲調整批次"],
    code: `# 示意：AIMD 調整批次大小
if last_latency < slo:
    batch_size += 1
else:
    batch_size = int(batch_size * 0.9)`,
    caseTitle: "線上預測系統",
    case: "Clipper 是早期的線上預測服務系統，提出依延遲目標動態調整批次大小的做法。",
    related: ["continuous-batching", "iteration-level-scheduling", "latency-throughput", "goodput"],
    sources: [["Clipper (USENIX NSDI 2017)", "https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw"]],
    papers: [p("Clipper: A Low-Latency Online Prediction Serving System", "NSDI", 2017, "https://arxiv.org/abs/1612.03079", "提出依延遲目標動態調整批次大小。")],
  }),
  c({
    id: "model-cold-start",
    title: "Model Cold Start",
    zh: "模型冷啟動",
    summary: "模型從未載入到可以服務第一個請求所需的時間，是自動擴縮時的主要延遲來源。",
    intuition: "工廠開機要先把機器預熱。訂單突然進來時，預熱太久就趕不上。",
    mechanism: [
      "時間花在下載權重、載入到 GPU，以及初始化與編譯。",
      "可以預先快取權重、使用 mmap，或保留一些暖機中的副本。",
      "縮容省錢，但下次擴容就要付出冷啟動成本。",
    ],
    diagram: ["請求抵達但無副本", "取得 GPU", "下載並載入權重", "開始服務"],
    code: `import time, torch
from transformers import AutoModelForCausalLM

t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")
print(f"載入耗時: {time.perf_counter() - t0:.1f}s")`,
    caseTitle: "Serverless 推論",
    case: "ServerlessLLM 研究 serverless 環境中的 LLM 載入延遲，提出利用本地儲存與快取加速模型啟動。",
    related: ["safetensors", "model-serving", "kubernetes", "model-multiplexing"],
    sources: [["ServerlessLLM (GitHub)", "https://github.com/ServerlessLLM/ServerlessLLM"]],
    papers: [p("ServerlessLLM: Low-Latency Serverless Inference for Large Language Models", "OSDI", 2024, "https://arxiv.org/abs/2401.14351", "分析並降低 LLM 的載入延遲。")],
  }),
  c({
    id: "safetensors",
    title: "Safetensors",
    zh: "Safetensors 權重格式",
    summary: "只儲存張量與中繼資料的權重格式，載入時不會執行任意程式碼，也能直接以 mmap 讀取。",
    intuition: "只裝貨物的貨櫃，打開時不會有機關跑出來；而且可以只取其中一箱，不必全部搬完。",
    mechanism: [
      "檔案開頭是 JSON 標頭，記錄每個張量的名稱、型別、形狀與位移。",
      "標頭之後就是原始資料，可以用 mmap 零複製存取。",
      "相較於使用 pickle 的權重格式，不會在載入時執行程式碼。",
    ],
    diagram: ["標頭長度", "JSON 標頭", "張量原始資料", "mmap 延遲讀取"],
    code: `from safetensors.torch import save_file, load_file

save_file({"weight": tensor}, "model.safetensors")
tensors = load_file("model.safetensors", device="cuda")`,
    caseTitle: "模型分享的預設格式",
    case: "Hugging Face 開發了 Safetensors，並在 Hub 與 Transformers 中廣泛使用。",
    related: ["model-cold-start", "gguf", "cpu-gpu-offloading", "model-serving"],
    sources: [["Hugging Face: Safetensors", "https://huggingface.co/docs/safetensors/index"]],
  }),
  c({
    id: "cpu-gpu-offloading",
    title: "CPU-GPU Offloading",
    zh: "CPU 與 GPU 分層載入",
    summary: "GPU 記憶體不足時，把部分權重放在 CPU 或磁碟，需要時才搬移，用速度換容量。",
    intuition: "桌面放不下所有書，就把一部分放書架，要用時再拿過來。放得下，但拿書要花時間。",
    mechanism: [
      "框架把模型層依序配置到 GPU、CPU 與磁碟。",
      "執行到某層時，才把該層的權重搬到 GPU。",
      "頻繁搬移會大幅拖慢速度，因此較適合批次任務或低流量場景。",
    ],
    diagram: ["GPU：常用層", "CPU：其餘層", "磁碟：溢出部分", "執行時搬移"],
    code: `from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    max_memory={0: "10GiB", "cpu": "32GiB"},
)`,
    caseTitle: "FlexGen",
    case: "FlexGen 研究在單張 GPU 上以權重與 KV Cache 分層儲存來執行大型模型，並著重在高吞吐量的批次場景。",
    related: ["kv-cache-offloading", "safetensors", "llama-cpp", "gpu-memory"],
    sources: [["Hugging Face Accelerate: Big model inference", "https://huggingface.co/docs/accelerate/usage_guides/big_modeling"]],
    papers: [p("FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU", "ICML", 2023, "https://arxiv.org/abs/2303.06865", "探討跨 GPU、CPU 與磁碟的資源配置。")],
  }),
  c({
    id: "semantic-caching",
    title: "Semantic Caching",
    zh: "語意快取",
    summary: "把意思相近的問題對應到已有的回答，命中時完全不必呼叫模型。",
    intuition: "客服發現新問題和常見問題意思一樣，就直接回覆標準答案，不用再寫一次。",
    mechanism: [
      "把問題轉成 embedding，並到向量資料庫中尋找相近的紀錄。",
      "相似度超過閾值就回傳先前的答案。",
      "閾值太低可能回傳錯誤答案，答案涉及時效或個人資料時也要特別小心。",
    ],
    diagram: ["問題 embedding", "向量搜尋", "相似度夠高則回傳快取", "否則呼叫模型並寫入"],
    code: `# 示意
vec = embed(question)
hit = index.search(vec, top_k=1)
if hit and hit.score > 0.9:
    return hit.answer
answer = llm(question)
index.add(vec, answer)`,
    caseTitle: "與前綴快取的差異",
    case: "前綴快取需要 token 完全相同，重用的是 KV Cache；語意快取比對意思，重用的是整個答案，因此有答錯的風險。GPTCache 是這類工具的開源例子。",
    related: ["prefix-caching", "embeddings", "vector-database", "model-routing"],
    sources: [["GPTCache (GitHub)", "https://github.com/zilliztech/GPTCache"]],
    papers: [p("GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching", "arXiv", 2024, "https://arxiv.org/abs/2411.05276", "以 embedding 快取評估成本與延遲的改善。")],
  }),
  c({
    id: "text-generation-inference",
    title: "Text Generation Inference (TGI)",
    zh: "TGI 推論伺服器",
    summary: "Hugging Face 開發的 LLM 推論伺服器，提供連續批次、串流輸出與量化等功能。",
    intuition: "現成的餐廳廚房系統，點餐、出餐和排程都已經配置好，只要把菜單（模型）放進去。",
    mechanism: [
      "以 Rust 撰寫路由與排程，模型執行則使用 Python。",
      "以動態批次把請求組合起來，並透過 SSE 串流輸出。",
      "提供指標與追蹤功能，方便部署到正式環境。",
    ],
    diagram: ["HTTP 請求", "Router 排程", "Model server", "GPU 生成", "串流輸出"],
    code: `docker run --gpus all -p 8080:80 \\
  ghcr.io/huggingface/text-generation-inference:latest \\
  --model-id <model>

curl localhost:8080/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"inputs":"你好","parameters":{"max_new_tokens":20}}'`,
    caseTitle: "Hugging Face 生態",
    case: "TGI 由 Hugging Face 開發，並開放原始碼，文件中說明了其架構與支援的量化方式。",
    related: ["iteration-level-scheduling", "continuous-batching", "vllm", "model-serving"],
    sources: [["Hugging Face: TGI", "https://huggingface.co/docs/text-generation-inference/index"]],
    papers: [ORCA_PAPER("TGI 使用的動態批次，概念上來自 iteration-level scheduling。")],
  }),
  c({
    id: "shortest-job-first-llm",
    title: "Shortest Job First for LLM",
    zh: "LLM 短作業優先",
    summary: "預測輸出長度並優先處理較短的請求，以降低平均等待時間。",
    intuition: "超市讓只買一瓶水的人先結帳，大家平均等得比較少，但要避免大量購物者永遠排不到。",
    mechanism: [
      "先來先服務容易讓短請求被長請求擋住（head-of-line blocking）。",
      "輸出長度事先未知，因此要用模型估算或排序。",
      "要處理飢餓問題，例如等待時間越久優先權越高。",
    ],
    diagram: ["請求佇列", "預測輸出長度", "依預測排序", "防止飢餓", "送入批次"],
    code: `# 示意
for req in waiting:
    req.score = predictor(req.prompt)   # 預測的相對長度
waiting.sort(key=lambda r: r.score - aging * r.wait_time)`,
    caseTitle: "學習排序",
    case: "相關研究指出，不必精確預測長度，只要能正確排序請求，就能在排程上獲得好處。",
    related: ["fair-llm-scheduling", "iteration-level-scheduling", "time-to-first-token", "goodput"],
    sources: [["Efficient LLM Scheduling by Learning to Rank (arXiv)", "https://arxiv.org/abs/2408.15792"]],
    papers: [p("Efficient LLM Scheduling by Learning to Rank", "NeurIPS", 2024, "https://arxiv.org/abs/2408.15792", "以排序學習近似短作業優先。")],
  }),
  c({
    id: "interference-isolation",
    title: "Prefill-Decode Interference",
    zh: "Prefill 與 Decode 干擾",
    summary: "運算密集的 prefill 與記憶體密集的 decode 混在一起時會互相拖慢，因此需要隔離或切分。",
    intuition: "同一條車道同時走重型貨車和機車，機車會被卡住，所以要分流或限制貨車長度。",
    mechanism: [
      "Prefill 一次處理很多 token，decode 則一次只處理一個 token。",
      "同批次執行時，decode 必須等 prefill 完成，形成明顯的延遲尖峰。",
      "解法包括切分 prefill、控制同時執行的量，或是分開部署。",
    ],
    diagram: ["Prefill 請求", "Decode 請求", "同批次造成延遲", "切分或分離部署"],
    code: `# 示意：控制每次迭代的 prefill 預算
budget = token_budget
for req in prefill_queue:
    n = min(req.remaining, budget)
    schedule_prefill_chunk(req, n)
    budget -= n
    if budget == 0:
        break`,
    caseTitle: "TetriInfer",
    case: "TetriInfer 研究 prefill 與 decode 之間的干擾，並主張把兩者分開，同時以固定大小的 chunk 處理 prefill。",
    related: ["chunked-prefill", "prefill-decode-disaggregation", "time-per-output-token", "fair-llm-scheduling"],
    sources: [["TetriInfer (arXiv)", "https://arxiv.org/abs/2401.11181"]],
    papers: [p("Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads", "arXiv", 2024, "https://arxiv.org/abs/2401.11181", "分析干擾來源，並提出分離式架構。")],
  }),
  c({
    id: "energy-per-token",
    title: "Energy per Token",
    zh: "每 Token 能耗",
    summary: "生成每個 token 所消耗的能量，是評估推論成本與資料中心效率的重要指標。",
    intuition: "像汽車的每公里油耗。開得快不一定省油，需要找到合適的速度。",
    mechanism: [
      "能耗由 GPU 功率與生成時間決定，兩者都受批次和頻率影響。",
      "批次較大通常可提高每瓦吞吐量，但也會增加延遲。",
      "調整 GPU 頻率與副本數，可以在符合 SLO 的前提下降低能耗。",
    ],
    diagram: ["量測 GPU 功率", "累積能耗（焦耳）", "除以輸出 token 數", "每 token 能耗"],
    code: `import pynvml

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)   # 毫焦耳
tokens = run_generation()
e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
print((e1 - e0) / 1000 / tokens, "J/token")`,
    caseTitle: "能耗感知的服務",
    case: "DynamoLLM 研究如何在符合延遲目標的前提下，調整模型副本與 GPU 頻率以節省能源。",
    related: ["latency-throughput", "goodput", "gpu-memory", "mlperf-inference"],
    sources: [["NVIDIA NVML API", "https://docs.nvidia.com/deploy/nvml-api/index.html"]],
    papers: [p("DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency", "HPCA", 2025, "https://arxiv.org/abs/2408.00741", "探討如何在維持 SLO 的同時節省能源。")],
  }),
  c({
    id: "mlperf-inference",
    title: "MLPerf Inference",
    zh: "MLPerf 推論效能基準",
    summary: "MLCommons 維護的業界效能基準，以統一的規則比較不同硬體與軟體的推論能力。",
    intuition: "統一賽道和規則的比賽，各家車隊的成績才有比較的意義。",
    mechanism: [
      "定義模型、資料集與品質目標，成績必須先達到準確率要求。",
      "以不同情境模擬真實負載，例如 offline、server 與 single stream。",
      "提交的結果會公開，並經過同業審查。",
    ],
    diagram: ["規定模型與資料", "LoadGen 產生負載", "量測延遲與吞吐量", "審查後公開"],
    code: `# 概念：Server 情境下，請求以 Poisson 分布抵達，
# 必須在達到延遲限制的前提下回報可支撐的 QPS
# 實際執行請參考 mlcommons/inference 的說明`,
    caseTitle: "硬體廠商的比較依據",
    case: "NVIDIA、Google、Intel、AMD 等廠商都曾提交 MLPerf Inference 結果，也納入了 LLM 相關的工作負載。",
    related: ["latency-throughput", "goodput", "inference-simulation", "energy-per-token"],
    sources: [["MLCommons: MLPerf Inference", "https://mlcommons.org/benchmarks/inference-datacenter/"], ["mlcommons/inference (GitHub)", "https://github.com/mlcommons/inference"]],
    papers: [p("MLPerf Inference Benchmark", "ISCA", 2020, "https://arxiv.org/abs/1911.02549", "說明 MLPerf Inference 的設計與量測方法。")],
  }),
  c({
    id: "inference-simulation",
    title: "Inference Simulation",
    zh: "推論系統模擬",
    summary: "不實際使用 GPU，以模擬器評估不同的設定，節省尋找最佳部署方式的成本。",
    intuition: "飛行員先在模擬艙練習，不必每種情況都真的把飛機開上天。",
    mechanism: [
      "以少量實測資料建立模型執行時間的預測。",
      "在模擬中重現排程、批次與記憶體的行為。",
      "可以快速比較副本數、批次策略與硬體，但結果仍需實際驗證。",
    ],
    diagram: ["實測少量資料", "訓練時間預測模型", "模擬排程與負載", "選出設定", "實機驗證"],
    code: `# Vidur 的使用方式請參考官方 README
python -m vidur.main \\
  --replica_config_device a100 \\
  --cluster_config_num_replicas 1`,
    caseTitle: "Vidur",
    case: "Vidur 是 Microsoft 開源的 LLM 推論模擬器，用來探索部署設定空間，論文也說明了其預測準確度。",
    related: ["goodput", "mlperf-inference", "latency-throughput", "chunked-prefill"],
    sources: [["Vidur (GitHub)", "https://github.com/microsoft/vidur"]],
    papers: [p("Vidur: A Large-Scale Simulation Framework For LLM Inference", "MLSys", 2024, "https://arxiv.org/abs/2405.05465", "以模擬探索 LLM 部署設定。")],
  }),
  c({
    id: "layerskip",
    title: "LayerSkip",
    zh: "LayerSkip 提前退出",
    summary: "訓練模型讓前面幾層也能輸出合理結果，生成時先用前幾層草擬，再以其餘層驗證。",
    intuition: "資淺同事先寫草稿，資深同事只檢查，不必每個字都由資深同事親自寫。",
    mechanism: [
      "訓練時對中間層也計算損失，並隨機略過後面的層。",
      "推論時以前面的層作為草稿模型，再由完整模型驗證。",
      "不需要額外的模型，且能共用部分計算結果。",
    ],
    diagram: ["前幾層產生草稿", "其餘層驗證", "接受正確 token", "重用已算的層"],
    code: `# 示意
draft = model(tokens, num_layers=early_exit)
logits = model.resume_from(early_exit, draft.hidden_states)
accepted = accept_matching(draft.tokens, logits)`,
    caseTitle: "Meta 的研究",
    case: "Meta 的 LayerSkip 論文展示這個方法用於推測解碼的可能性，並提供了訓練好的模型與實作。",
    related: ["speculative-decoding", "medusa-decoding", "eagle-decoding", "knowledge-distillation"],
    sources: [["LayerSkip (GitHub)", "https://github.com/facebookresearch/LayerSkip"]],
    papers: [p("Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding", "ACL", 2024, "https://arxiv.org/abs/2404.16710", "讓模型自行作為自己的草稿模型。")],
  }),
  c({
    id: "half-precision-inference",
    title: "Half-Precision Inference",
    zh: "半精度推論",
    summary: "以 FP16 或 BF16 取代 FP32 進行推論，權重記憶體可減半，也常能提高速度。",
    intuition: "記帳時只寫到小數點後幾位。大多數情況夠用，也省下不少紙和時間。",
    mechanism: [
      "FP16 動態範圍小，容易溢位；BF16 保留 FP32 的指數位數，但精度較低。",
      "GPU 通常有針對半精度最佳化的運算。",
      "數值敏感的運算可能仍需以較高精度計算。",
    ],
    diagram: ["FP32 權重", "轉成 FP16 / BF16", "推論", "必要時局部提高精度"],
    code: `import torch

model = model.to(torch.bfloat16).cuda().eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(**inputs)`,
    caseTitle: "最先嘗試的省記憶體方法",
    case: "許多開源 LLM 都以 BF16 或 FP16 發布權重，並建議以對應的精度載入。",
    related: ["mixed-precision", "quantization", "tensor-cores", "gpu-memory"],
    sources: [["PyTorch: Automatic Mixed Precision", "https://pytorch.org/docs/stable/amp.html"]],
    papers: [p("Mixed Precision Training", "ICLR", 2018, "https://arxiv.org/abs/1710.03740", "說明在深度學習中使用半精度的方法。")],
  }),
  c({
    id: "hardware-aware-nas",
    title: "Hardware-Aware NAS",
    zh: "硬體感知架構搜尋",
    summary: "搜尋模型架構時把目標硬體的延遲或能耗列入考量，找出適合該裝置的模型。",
    intuition: "買車時不只看馬力，還要考慮家裡的車庫大小和當地路況。",
    mechanism: [
      "以查表或模型估計各種架構在目標硬體上的延遲。",
      "同時以準確率與延遲作為搜尋目標。",
      "Once-for-All 先訓練大型超網路，再從中挑選子網路，不必為每個裝置重新訓練。",
    ],
    diagram: ["搜尋空間", "超網路訓練", "延遲預測", "挑選子網路", "部署"],
    code: `# 示意
best = None
for arch in sample_architectures():
    if latency_lut(arch) <= target_ms:
        acc = evaluate(supernet.subnet(arch))
        best = max(best, (acc, arch), key=lambda x: x[0]) if best else (acc, arch)`,
    caseTitle: "Once-for-All",
    case: "MIT 提出的 Once-for-All 把訓練與搜尋分開，讓同一個超網路可以對應多種硬體條件。",
    related: ["structured-pruning", "weight-pruning", "quantization", "knowledge-distillation"],
    sources: [["Once-for-All (GitHub)", "https://github.com/mit-han-lab/once-for-all"]],
    papers: [p("Once-for-All: Train One Network and Specialize it for Efficient Deployment", "ICLR", 2020, "https://arxiv.org/abs/1908.09791", "把訓練與架構搜尋分開。")],
  }),
  c({
    id: "kv-cache-streaming",
    title: "KV Cache Streaming",
    zh: "KV Cache 串流傳輸",
    summary: "把 KV Cache 編碼壓縮後透過網路傳送，讓遠端的上下文載入比重新計算更快。",
    intuition: "與其現場重新做菜，不如把半成品真空包裝快遞過來，收到後加熱就能吃。",
    mechanism: [
      "把 KV Cache 視為張量，並以類似影片編碼的方式壓縮。",
      "分成多段傳送，並依網路狀況調整壓縮等級。",
      "若傳輸不及，可退回傳送文字並重新計算。",
    ],
    diagram: ["遠端 KV Cache", "分段與編碼", "網路傳輸", "解碼後載入 GPU"],
    code: `# 示意
for chunk in split(kv_cache):
    level = choose_level(bandwidth(), deadline)
    send(encode(chunk, level))`,
    caseTitle: "CacheGen",
    case: "CacheGen 研究如何以編碼與串流的方式傳送 KV Cache，以降低載入長上下文的延遲。",
    related: ["kv-cache-offloading", "rdma-kv-transfer", "prefix-caching", "time-to-first-token"],
    sources: [["CacheGen (arXiv)", "https://arxiv.org/abs/2310.07240"]],
    papers: [p("CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving", "SIGCOMM", 2024, "https://arxiv.org/abs/2310.07240", "提出 KV Cache 的編碼與串流方法。")],
  }),
  c({
    id: "rdma-kv-transfer",
    title: "RDMA KV Cache Transfer",
    zh: "RDMA KV Cache 傳輸",
    summary: "透過 RDMA 在機器之間快速搬移 KV Cache，是 prefill 與 decode 分離部署的關鍵。",
    intuition: "兩個倉庫之間有專用傳送帶，貨物直接送到位，不必層層經手。",
    mechanism: [
      "Prefill 節點完成後，把 KV Cache 傳給 decode 節點。",
      "RDMA 繞過 CPU 與作業系統的網路堆疊，能降低延遲與 CPU 負擔。",
      "傳輸時間會直接影響 TTFT 與 TPOT，因此要與排程一起規劃。",
    ],
    diagram: ["Prefill 節點", "註冊記憶體", "RDMA 傳輸 KV", "Decode 節點繼續生成"],
    code: `# 示意
region = register_memory(kv_cache)
rdma_write(decode_node, region, remote_addr)
notify(decode_node, request_id)`,
    caseTitle: "Mooncake Transfer Engine",
    case: "Mooncake 專案開源了 Transfer Engine，用於在 DRAM、VRAM 與 NVMe 之間搬移資料，並支援 RDMA 等傳輸方式。",
    related: ["prefill-decode-disaggregation", "kv-cache-offloading", "kv-cache-streaming", "time-per-output-token"],
    sources: [["Mooncake (GitHub)", "https://github.com/kvcache-ai/Mooncake"]],
    papers: [MOONCAKE_PAPER("以 KV Cache 為中心的分離式架構與傳輸設計。")],
  }),
  c({
    id: "model-multiplexing",
    title: "Model Multiplexing",
    zh: "多模型共用 GPU",
    summary: "讓多個模型共用同一批 GPU，依需求切換或切分，以提高低流量模型的資源使用率。",
    intuition: "多家小店共用一間廚房。每家單獨開廚房太貴，共用則要安排好時段。",
    mechanism: [
      "多數模型流量不高，獨佔 GPU 容易造成浪費。",
      "把模型切分到多張 GPU，並讓不同模型共用裝置，可以提高總容量。",
      "切換模型有載入成本，因此要與流量預測一起考量。",
    ],
    diagram: ["多個模型", "切分與配置", "共用 GPU 群組", "依請求排程"],
    code: `# 示意
placement = planner.place(models, gpus, slo)
for req in requests:
    group = placement.lookup(req.model)
    group.submit(req)`,
    caseTitle: "AlpaServe",
    case: "AlpaServe 研究多模型服務，指出適當的切分方式能在同樣的 GPU 上服務更高的流量。",
    related: ["model-serving", "model-cold-start", "tensor-parallelism", "goodput"],
    sources: [["AlpaServe (arXiv)", "https://arxiv.org/abs/2302.11665"]],
    papers: [p("AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving", "OSDI", 2023, "https://arxiv.org/abs/2302.11665", "結合模型切分與統計多工。")],
  }),
  c({
    id: "live-migration-llm",
    title: "LLM Live Migration",
    zh: "LLM 請求即時遷移",
    summary: "把進行中的請求連同 KV Cache 移到另一個副本，在不中斷服務的情況下重新平衡負載。",
    intuition: "乘客不用下車重新排隊，車輛可以在行駛中換到另一條比較順的車道。",
    mechanism: [
      "輸出長度難以預測，副本的負載會隨時間變得不平均。",
      "遷移時先持續複製 KV Cache，最後只需很短的暫停。",
      "全域排程器依負載決定遷移時機，也可用於隔離高優先權請求或維護。",
    ],
    diagram: ["過載副本", "複製 KV Cache", "短暫切換", "目標副本繼續生成"],
    code: `# 示意
while dirty_blocks(req) > threshold:
    copy_blocks(req, source, target)
pause(req)
copy_blocks(req, source, target)
resume(req, on=target)`,
    caseTitle: "Llumnix",
    case: "Llumnix 提出 LLM 請求的即時遷移，並以此作為叢集排程的基本機制。",
    related: ["cache-aware-routing", "request-preemption", "kv-cache", "fair-llm-scheduling"],
    sources: [["Llumnix (GitHub)", "https://github.com/AlibabaPAI/llumnix"]],
    papers: [p("Llumnix: Dynamic Scheduling for Large Language Model Serving", "OSDI", 2024, "https://arxiv.org/abs/2406.03243", "提出請求即時遷移與動態排程。")],
  }),
  c({
    id: "copy-on-write-kv-cache",
    title: "Copy-on-Write KV Cache",
    zh: "寫入時複製的 KV Cache",
    summary: "讓多個輸出序列共用相同的 KV Cache 區塊，只有內容開始不同時才複製，以節省記憶體。",
    intuition: "多人共讀同一份文件，只有想修改的人才另外影印一份。",
    mechanism: [
      "平行取樣與 beam search 的多個序列，開頭都來自同一個 prompt。",
      "KV 區塊維護參考計數，多個序列可以指向同一區塊。",
      "要寫入共用區塊時才複製，避免一開始就保存多份。",
    ],
    diagram: ["共同 prompt 的 KV 區塊", "多序列共用", "寫入時檢查參考計數", "必要時複製"],
    code: `# 示意
def append(seq, k, v):
    block = seq.last_block
    if block.ref_count > 1:
        block.ref_count -= 1
        block = block.copy()
        seq.last_block = block
    block.write(k, v)`,
    caseTitle: "vLLM 的平行取樣",
    case: "vLLM 論文說明如何以分頁與 copy-on-write 機制，讓平行取樣與 beam search 共用 KV Cache。",
    related: ["paged-attention", "vllm", "kv-cache", "sampling-decoding"],
    sources: [VLLM_DOCS],
    papers: [VLLM_PAPER("說明分頁式 KV Cache 上的共用與 copy-on-write。")],
  }),
];
