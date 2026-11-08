import type { Concept } from "@/types";

export const llmAgentConcepts: Concept[] = [
  {
    id: "rag",
    title: "Retrieval-Augmented Generation",
    zh: "檢索增強生成",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "先從外部知識庫檢索與問題相關的內容，再把檢索結果放入提示中交給語言模型作答，使回答能依據最新或私有的資料。",
    intuition:
      "像開書考試：不必把所有知識背起來，遇到問題先翻參考書找出相關章節，再根據那幾頁寫出答案。",
    mechanism: [
      "離線階段：把文件切塊（chunking），以 embedding 模型轉成向量，並寫入向量索引或搜尋引擎。",
      "線上階段：把使用者問題轉成 embedding 或查詢，檢索出最相關的若干文件片段，必要時再做重排序（rerank）。",
      "把檢索到的片段與問題組合成提示，要求模型根據資料回答並可標註引用來源。",
      "品質取決於檢索與生成兩端：檢索不到、檢索錯或上下文放置不當，都會造成錯誤答案，因此需要分別評估檢索品質與最終回答的忠實度。",
    ],
    diagram: [
      "文件切塊並建立向量索引",
      "使用者問題",
      "檢索 top-k 片段（可重排序）",
      "組合提示（問題 + 片段）",
      "LLM 生成並引用來源",
    ],
    code: `# 概念性流程
docs = retriever.search(question, k=5)
context = "\\n\\n".join(d.text for d in docs)
prompt = f"請根據以下資料回答，找不到就說不知道。\\n{context}\\n\\n問題：{question}"
answer = llm.generate(prompt)`,
    caseTitle: "RAG 原始論文與綜述",
    case: "Lewis 等人的論文提出結合參數化記憶與非參數化檢索的 RAG 架構，後續有綜述整理了「Naive、Advanced、Modular」等 RAG 設計模式。實際系統的做法差異很大，需要用自己的資料評估。",
    related: ["embeddings", "vector-database", "chunking", "reranking"],
    sources: [
      ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "https://arxiv.org/abs/2005.11401"],
      ["Retrieval-Augmented Generation for LLMs: A Survey", "https://arxiv.org/abs/2312.10997"],
    ],
  },
  {
    id: "vector-database",
    title: "Vector Database",
    zh: "向量資料庫",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "專門儲存向量並支援近似最近鄰（ANN）搜尋的資料系統，用於語意檢索與 RAG。",
    intuition:
      "像一座依「意思相近」而非「字母順序」排列的圖書館，問一個問題就能找出最相近的幾本書。",
    mechanism: [
      "每筆資料包含向量與可選的中繼資料（metadata）與原文。",
      "為了避免與所有向量逐一比對，系統建立近似最近鄰索引，例如 HNSW、IVF 或乘積量化（PQ），以犧牲少量召回率換取速度。",
      "查詢時對查詢向量搜尋 top-k 最相近者，距離度量常用餘弦相似度、內積或 L2，並可結合 metadata 過濾。",
      "選型時要考慮資料量、更新頻率、過濾能力、記憶體或磁碟成本與召回率；不同 embedding 模型產生的向量不能混用。",
    ],
    diagram: [
      "文件 → embedding 向量",
      "寫入向量與 metadata",
      "建立 ANN 索引",
      "查詢向量 top-k 搜尋",
      "回傳相近項目與 metadata",
    ],
    code: `import faiss, numpy as np

d = 128
xb = np.random.rand(10000, d).astype("float32")
index = faiss.IndexFlatIP(d)       # 精確內積搜尋；規模大時改用 HNSW / IVF
index.add(xb)
D, I = index.search(xb[:3], 5)`,
    caseTitle: "FAISS",
    case: "Meta 公開了 FAISS 函式庫及其論文，說明了大規模向量相似度搜尋常用的索引與量化方法。市面上的向量資料庫在功能、規模與託管方式上各有不同，需依需求評估。",
    related: ["embeddings", "hnsw", "rag", "hybrid-search"],
    sources: [
      ["The Faiss library", "https://arxiv.org/abs/2401.08281"],
      ["Faiss wiki", "https://github.com/facebookresearch/faiss/wiki"],
    ],
  },
  {
    id: "hnsw",
    title: "HNSW",
    zh: "階層式可導航小世界圖",
    category: "LLM & Agents",
    level: "Reference",
    status: "published",
    summary:
      "一種以多層鄰近圖組織向量的近似最近鄰索引，搜尋時由上層粗略導航到下層精確搜尋，查詢速度與召回率都很高。",
    intuition:
      "像在城市找地址：先看高速公路地圖跳到大致區域，再換成街道地圖逐步靠近目標。",
    mechanism: [
      "每個向量是圖中的節點，與若干鄰近節點相連；節點被隨機分配到不同層級，越高層的節點越少。",
      "搜尋時從最高層的入口點開始，貪婪地移動到更接近查詢向量的鄰居，直到無法改善，再往下一層以前一層的結果為起點繼續。",
      "在最底層以較大的候選集合（參數 ef）做更細緻的搜尋，ef 越大召回率越高但越慢。",
      "建構參數（如每個節點的連接數 M、建構時的 ef）會影響索引大小、建構時間與召回；索引常放在記憶體中，大規模資料的記憶體成本需要評估。",
    ],
    diagram: [
      "查詢向量",
      "頂層入口點",
      "逐層貪婪導航",
      "底層 ef 候選搜尋",
      "回傳 top-k",
    ],
    code: `import hnswlib, numpy as np

d, n = 64, 10000
data = np.random.rand(n, d).astype("float32")
p = hnswlib.Index(space="cosine", dim=d)
p.init_index(max_elements=n, ef_construction=200, M=16)
p.add_items(data)
p.set_ef(50)
labels, dist = p.knn_query(data[:2], k=5)`,
    caseTitle: "HNSW 論文",
    case: "Malkov 與 Yashunin 的論文提出 HNSW，並與當時其他 ANN 方法比較。許多向量搜尋系統提供 HNSW 索引，參數設定與效果須依資料集調整。",
    related: ["vector-database", "embeddings", "rag", "hybrid-search"],
    sources: [
      ["Efficient and robust approximate nearest neighbor search using HNSW", "https://arxiv.org/abs/1603.09320"],
      ["The Faiss library", "https://arxiv.org/abs/2401.08281"],
    ],
  },
  {
    id: "chunking",
    title: "Chunking",
    zh: "文件切塊",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "把長文件切成適當大小的片段後再建立索引，使檢索回來的內容既聚焦又保有足夠的上下文。",
    intuition:
      "像把一本書拆成便利貼：太大的便利貼塞了太多無關資訊，太小的又缺少前後文，大小要剛好。",
    mechanism: [
      "常見策略包含固定字元或 token 數切分、依段落與標題的結構化切分，以及依語意邊界的切分。",
      "相鄰片段常設定重疊（overlap），避免重要資訊被切在邊界上。",
      "片段太大會稀釋相關性並佔用上下文視窗，太小則缺少脈絡；最佳大小取決於文件型態與問題類型。",
      "也可以保存每個片段的來源、標題階層等 metadata，檢索後用於引用、過濾或擴展周圍上下文。",
    ],
    diagram: [
      "原始文件",
      "清理並依結構切分",
      "設定大小與重疊",
      "附加 metadata",
      "embedding 並寫入索引",
    ],
    code: `def chunk(text, size=500, overlap=50):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out`,
    caseTitle: "RAG 綜述與長上下文研究",
    case: "RAG 綜述整理了切塊與索引相關的設計選擇；「Lost in the Middle」研究則顯示模型對放在長上下文中間的資訊利用較差，提醒不能只靠塞更多內容。切塊策略需要以檢索評估實測。",
    related: ["rag", "embeddings", "context-window", "reranking"],
    sources: [
      ["Retrieval-Augmented Generation for LLMs: A Survey", "https://arxiv.org/abs/2312.10997"],
      ["Lost in the Middle", "https://arxiv.org/abs/2307.03172"],
    ],
  },
  {
    id: "reranking",
    title: "Reranking",
    zh: "重排序",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "在初步檢索得到的候選集合上，用更精細（通常也更慢）的模型重新評分並排序，提升最前面結果的相關性。",
    intuition:
      "像先由海選粗篩出五十位候選人，再請資深面試官逐一仔細面談，挑出最適合的前幾名。",
    mechanism: [
      "第一階段以向量檢索或關鍵字檢索快速取回數十到數百個候選（重視召回率）。",
      "第二階段使用 cross-encoder 等模型，把查詢與每個候選文件一起輸入，直接輸出相關性分數。",
      "因為每個候選都要跑一次模型，成本較高，所以只對少量候選做重排，再取前幾名交給 LLM。",
      "cross-encoder 通常比 bi-encoder 精確，但無法預先計算文件向量；需要在延遲與品質之間取捨。",
    ],
    diagram: [
      "查詢",
      "第一階段檢索 top-100",
      "Cross-encoder 逐一評分",
      "依分數重新排序",
      "取前幾名給 LLM",
    ],
    code: `from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = ce.predict([(query, d) for d in candidates])
ranked = [d for _, d in sorted(zip(scores, candidates), reverse=True)]`,
    caseTitle: "BERT 重排序",
    case: "「Passage Re-ranking with BERT」論文展示了以 BERT 做 cross-encoder 重排序可顯著提升段落檢索表現；Sentence-Transformers 文件說明了 cross-encoder 的用法。",
    related: ["rag", "hybrid-search", "vector-database", "embeddings"],
    sources: [
      ["Passage Re-ranking with BERT", "https://arxiv.org/abs/1901.04085"],
      ["Sentence-Transformers cross-encoders", "https://www.sbert.net/examples/applications/cross-encoder/README.html"],
    ],
  },
  {
    id: "hybrid-search",
    title: "Hybrid Search",
    zh: "混合檢索",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "結合關鍵字檢索（如 BM25）與向量語意檢索，並融合兩者的排名，兼顧精確詞彙匹配與語意相似。",
    intuition:
      "像找資料時同時用「關鍵字搜尋」和「請懂行的朋友推薦」，兩邊的建議合併起來通常比單用一種更可靠。",
    mechanism: [
      "稀疏檢索（BM25）擅長精確詞彙、專有名詞、型號與代碼，但對同義改寫較弱。",
      "稠密檢索（embedding）擅長語意相近的改寫與跨語言，但對罕見詞或精確字串可能失手。",
      "兩者各自取回候選後，用分數正規化加權，或用 Reciprocal Rank Fusion（RRF）依排名融合，不需要對齊不同分數的尺度。",
      "融合後常再接 reranker；權重與候選數需要用標註好的查詢集評估，並非越複雜越好。",
    ],
    diagram: [
      "查詢",
      "BM25 檢索",
      "向量檢索",
      "RRF 或加權融合",
      "（可選）重排序",
    ],
    code: `def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:               # 每個 ranking 是依序的文件 id 清單
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)`,
    caseTitle: "DPR 與 RRF",
    case: "Dense Passage Retrieval 論文研究了稠密檢索並與 BM25 比較，Cormack 等人的論文提出了 Reciprocal Rank Fusion 以融合多個排名。兩者都是常見的混合檢索元件。",
    related: ["vector-database", "reranking", "rag", "embeddings"],
    sources: [
      ["Dense Passage Retrieval for Open-Domain QA", "https://arxiv.org/abs/2004.04906"],
      ["Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods", "https://dl.acm.org/doi/10.1145/1571941.1572114"],
    ],
  },
  {
    id: "prompt-engineering",
    title: "Prompt Engineering",
    zh: "提示工程",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "設計與迭代給語言模型的指令、範例與格式，使其更穩定地產生符合需求的輸出。",
    intuition:
      "像給一位能力很強但不了解你情境的新同事寫工作說明：目標、背景、格式與範例講得越清楚，成果越接近你要的。",
    mechanism: [
      "清楚說明任務、角色、受眾、限制與期望的輸出格式，並提供必要的背景資料。",
      "提供少量範例（few-shot）示範輸入與輸出的對應，能有效引導風格與格式；GPT-3 論文把這類現象稱為 in-context learning。",
      "對需要多步驟推理的任務，可以要求模型逐步思考，或把任務拆成多個提示串接。",
      "把提示當成可測試的程式：建立評估集、比較不同版本、記錄結果，並留意模型更新後行為可能改變。",
    ],
    diagram: [
      "定義任務與成功標準",
      "撰寫提示（指令、範例、格式）",
      "在評估集上測試",
      "分析失敗案例",
      "調整提示並重新測試",
    ],
    code: `prompt = """你是客服摘要助理。
請把下列對話摘要成三點，使用繁體中文，每點不超過 30 字。

對話：
{conversation}
"""`,
    caseTitle: "模型廠商的官方指南",
    case: "Anthropic 與其他模型廠商都公開了提示撰寫指南，內容包含清楚指令、提供範例與結構化輸入等建議。不同模型對提示的反應不同，應以官方文件與自己的評估為準。",
    related: ["chain-of-thought", "context-window", "llm-as-judge", "function-calling"],
    sources: [
      ["Anthropic prompt engineering overview", "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview"],
      ["Language Models are Few-Shot Learners", "https://arxiv.org/abs/2005.14165"],
    ],
  },
  {
    id: "chain-of-thought",
    title: "Chain-of-Thought",
    zh: "思維鏈",
    category: "LLM & Agents",
    level: "Reference",
    status: "published",
    summary:
      "讓模型在給出最終答案前先寫出中間推理步驟，提升算術、常識與符號推理等多步驟任務的表現。",
    intuition:
      "像解數學題時寫出計算過程，而不是直接報答案：步驟攤開來，出錯的機會較少，也比較好檢查。",
    mechanism: [
      "在 few-shot 範例中示範「問題 → 推理步驟 → 答案」，模型便會模仿以逐步推理的方式回答。",
      "Kojima 等人發現，即使不給範例，只在提示中要求逐步思考（zero-shot CoT），在某些任務上也能改善結果。",
      "效果通常在模型規模夠大時才明顯，對簡單任務幫助有限，並增加輸出 token 數與延遲。",
      "寫出的推理未必真實反映模型內部的計算過程，因此不能把它當成保證正確的證明，仍需驗證最終答案。",
    ],
    diagram: [
      "問題",
      "提示要求逐步推理",
      "模型產生中間步驟",
      "得出答案",
      "驗證最終答案",
    ],
    code: `prompt = """問題：商店有 23 顆蘋果，賣出 9 顆，又進貨 12 顆，現在有幾顆？
請一步一步思考，最後以「答案：」開頭給出結果。"""`,
    caseTitle: "Wei 等人與 Kojima 等人的研究",
    case: "Wei 等人的論文展示了思維鏈提示在多種推理基準上的改善，Kojima 等人則展示了零樣本版本。實際效益依模型與任務不同，需用自己的評估集驗證。",
    related: ["prompt-engineering", "react-agent", "sampling-decoding", "llm-as-judge"],
    sources: [
      ["Chain-of-Thought Prompting Elicits Reasoning", "https://arxiv.org/abs/2201.11903"],
      ["Large Language Models are Zero-Shot Reasoners", "https://arxiv.org/abs/2205.11916"],
    ],
  },
  {
    id: "function-calling",
    title: "Function Calling and Tool Use",
    zh: "函式呼叫與工具使用",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "讓模型依據工具的結構化描述，產生要呼叫哪個工具及其參數，由應用程式實際執行後再把結果回傳給模型。",
    intuition:
      "像秘書不會自己變出天氣預報，而是說「請幫我查天氣，地點是台北」，由你去查，再把結果帶回去繼續寫報告。",
    mechanism: [
      "開發者以 JSON Schema 等格式描述可用的工具：名稱、用途與參數。",
      "模型在回應中決定是否呼叫工具，並輸出工具名稱與符合 schema 的參數，而不是自己執行。",
      "應用程式負責驗證參數、實際執行工具（呼叫 API、查資料庫等），並把結果以工具結果訊息放回對話。",
      "模型讀取結果後繼續回答或再次呼叫工具；因為模型輸出可能格式不正確或被誘導，必須驗證參數、限制權限並對有副作用的操作設置確認機制。",
    ],
    diagram: [
      "提供工具描述與使用者問題",
      "模型輸出工具呼叫與參數",
      "應用程式驗證並執行工具",
      "回傳工具結果給模型",
      "模型產生最終回答",
    ],
    code: `tools = [{
    "name": "get_weather",
    "description": "查詢城市的目前天氣",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]
# 把 tools 連同訊息送給模型；收到 tool_use 後執行並回傳 tool_result`,
    caseTitle: "Toolformer 與官方 API 文件",
    case: "Toolformer 論文探討語言模型學習何時及如何呼叫外部工具；各模型廠商的 API 文件則說明了各自的工具使用格式與流程，格式不完全一致，需以所用服務的文件為準。",
    related: ["react-agent", "model-context-protocol", "prompt-injection", "prompt-engineering"],
    sources: [
      ["Toolformer", "https://arxiv.org/abs/2302.04761"],
      ["Anthropic tool use overview", "https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview"],
    ],
  },
  {
    id: "react-agent",
    title: "ReAct Agent",
    zh: "ReAct 代理架構",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "讓模型交錯進行推理（Thought）與行動（Action），根據行動後的觀察（Observation）調整下一步，直到完成任務。",
    intuition:
      "像偵探辦案：先想「下一步該查什麼」，去查，看到結果後再想下一步，而不是一開始就憑空下結論。",
    mechanism: [
      "迴圈由三種內容組成：Thought（推理）、Action（呼叫工具，如搜尋或計算）、Observation（工具回傳結果）。",
      "每一輪把累積的歷史放入提示，模型決定下一個 Thought 與 Action，應用程式執行後把 Observation 附加進去。",
      "當模型判斷已有足夠資訊時，輸出最終答案並結束迴圈。",
      "需要設定最大步數與成本上限，避免無限迴圈；工具失敗、資訊矛盾與提示被注入的內容誤導，是實務中常見的失敗來源。",
    ],
    diagram: [
      "任務",
      "Thought：決定下一步",
      "Action：呼叫工具",
      "Observation：取得結果",
      "重複直到完成並輸出答案",
    ],
    code: `history = [f"問題：{question}"]
for _ in range(MAX_STEPS):
    step = llm.generate("\\n".join(history))   # 產生 Thought 與 Action 或最終答案
    if step.is_final:
        return step.answer
    obs = run_tool(step.action, step.args)
    history += [step.text, f"Observation: {obs}"]`,
    caseTitle: "ReAct 論文",
    case: "Yao 等人的 ReAct 論文在問答與互動決策任務上展示了交錯推理與行動的效果。現今許多代理框架以類似迴圈為基礎，但可靠度仍需依任務設計評估與防護。",
    related: ["function-calling", "chain-of-thought", "model-context-protocol", "prompt-injection"],
    sources: [
      ["ReAct: Synergizing Reasoning and Acting", "https://arxiv.org/abs/2210.03629"],
      ["Toolformer", "https://arxiv.org/abs/2302.04761"],
    ],
  },
  {
    id: "model-context-protocol",
    title: "Model Context Protocol",
    zh: "模型上下文協定（MCP）",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "開放協定，標準化 AI 應用如何連接外部的工具、資料與提示，讓同一個 MCP 伺服器能被不同的客戶端重用。",
    intuition:
      "像 AI 世界的通用插座：工具只要做成符合規格的插頭，各種 AI 應用都能插上使用，不必為每個應用重做一次整合。",
    mechanism: [
      "MCP 採用客戶端—伺服器架構：AI 應用（host）內含 MCP 客戶端，連到一個或多個 MCP 伺服器。",
      "伺服器可以提供工具（tools，可被模型呼叫）、資源（resources，可讀取的資料）與提示模板（prompts）。",
      "通訊以 JSON-RPC 為基礎，支援本機（stdio）與遠端（HTTP）等傳輸方式。",
      "因為伺服器能代表使用者執行動作或存取資料，需要仔細處理授權、權限範圍與提示注入風險，並只連接信任的伺服器。",
    ],
    diagram: [
      "AI 應用（host）",
      "MCP 客戶端",
      "JSON-RPC 連線",
      "MCP 伺服器（工具／資源／提示）",
      "外部系統（API、資料庫、檔案）",
    ],
    code: `# 概念性：列出伺服器提供的工具並呼叫（SDK 介面以官方文件為準）
tools = await session.list_tools()
result = await session.call_tool("search_docs", {"query": "RAG"})`,
    caseTitle: "開放規格與官方 SDK",
    case: "MCP 的官方網站公開了規格、概念說明與多種語言的 SDK。支援狀況與規格版本持續演進，實作前請查閱最新文件。",
    related: ["function-calling", "react-agent", "prompt-injection", "context-window"],
    sources: [
      ["Model Context Protocol", "https://modelcontextprotocol.io/"],
      ["MCP specification", "https://modelcontextprotocol.io/specification"],
    ],
  },
  {
    id: "llm-as-judge",
    title: "LLM-as-a-Judge",
    zh: "以 LLM 擔任評審",
    category: "LLM & Agents",
    level: "Reference",
    status: "published",
    summary:
      "以一個強大的語言模型依據評分準則為其他模型的輸出打分或做成對比較，作為人工評估的低成本替代或補充。",
    intuition:
      "像請一位資深評審先看過評分標準，再替大量作品打分；速度快，但評審本人也有偏好，需要先校準。",
    mechanism: [
      "設計明確的評分準則（rubric）與輸出格式，讓評審模型對單一回答評分，或對兩個回答比較誰較好。",
      "MT-Bench 研究分析了這類評審與人類偏好的一致程度，也指出已知偏差，例如位置偏差（偏好先出現的答案）與冗長偏差（偏好較長的答案）。",
      "常見緩解做法包含交換回答順序重複評分、提供參考答案、要求先給理由再給分數。",
      "應定期抽樣與人工標註比對以校準；評審模型不是真相來源，對高風險或專業領域的結果仍需人工檢查。",
    ],
    diagram: [
      "定義評分準則",
      "評審模型打分或成對比較",
      "交換順序重複以降低偏差",
      "與人工標註抽樣比對",
      "彙整指標並迭代",
    ],
    code: `judge_prompt = """請依下列準則為回答評分（1-5）：正確性、完整性、清晰度。
先寫簡短理由，最後一行輸出「分數：N」。

問題：{q}
回答：{a}"""`,
    caseTitle: "MT-Bench 與 Chatbot Arena",
    case: "Zheng 等人的論文提出 MT-Bench 與 Chatbot Arena，並分析強模型作為評審與人類偏好的一致性及其偏差。結果依評審模型與任務而定。",
    related: ["ml-evaluation", "reward-model", "prompt-engineering", "chain-of-thought"],
    sources: [
      ["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "https://arxiv.org/abs/2306.05685"],
      ["Language Models are Few-Shot Learners", "https://arxiv.org/abs/2005.14165"],
    ],
  },
  {
    id: "context-window",
    title: "Context Window",
    zh: "上下文視窗",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "模型單次請求中可處理的 token 上限，包含輸入提示與生成的輸出；超出的內容需要截斷、摘要或檢索。",
    intuition:
      "像一張固定大小的工作桌，桌面能攤開的資料有限；要放新資料就得收掉舊的，或只放最重要的部分。",
    mechanism: [
      "視窗長度以 token 計，輸入與輸出共用額度；成本、延遲與記憶體用量都隨上下文長度增加。",
      "Attention 的計算量與 KV Cache 的大小隨序列長度成長，所以長上下文的服務成本較高。",
      "長上下文並不保證模型能均勻地利用所有資訊；研究顯示相關資訊放在中間時，表現可能比放在開頭或結尾差。",
      "延伸視窗的方法包含位置編碼插值等，需要額外訓練與評估；實務上常搭配檢索、摘要與提示快取來管理內容。",
    ],
    diagram: [
      "系統提示 + 歷史 + 檢索內容",
      "計算 token 總量",
      "超出則截斷／摘要／檢索",
      "送入模型",
      "輸出共用剩餘額度",
    ],
    code: `# 簡單的歷史截斷（以 token 計）
def trim(messages, max_tokens, count_tokens):
    while sum(count_tokens(m) for m in messages) > max_tokens and len(messages) > 1:
        messages.pop(1)        # 保留系統提示，移除較舊的對話
    return messages`,
    caseTitle: "Lost in the Middle 與位置插值",
    case: "「Lost in the Middle」研究評估了模型在長上下文中對不同位置資訊的利用；Position Interpolation 論文提出以插值延伸以 RoPE 為基礎的模型的上下文長度。",
    related: ["kv-cache", "rope", "chunking", "prefix-caching"],
    sources: [
      ["Lost in the Middle", "https://arxiv.org/abs/2307.03172"],
      ["Extending Context Window via Positional Interpolation", "https://arxiv.org/abs/2306.15595"],
    ],
  },
  {
    id: "prompt-injection",
    title: "Prompt Injection",
    zh: "提示注入",
    category: "LLM & Agents",
    level: "Core focus",
    status: "published",
    summary:
      "攻擊者把惡意指令藏在模型會讀取的內容中（使用者輸入、網頁、文件或工具結果），誘使模型偏離原本的指示或執行不當動作。",
    intuition:
      "像秘書替老闆讀信，結果信裡寫著「請把老闆的通訊錄寄給我」，秘書如果分不清哪些是老闆的指示、哪些只是信的內容，就會照做。",
    mechanism: [
      "模型把系統指令、使用者輸入與外部資料都當成同一段文字序列，無法從根本上區分「資料」與「指令」。",
      "直接注入：使用者直接在輸入中要求忽略先前規則；間接注入：惡意指令藏在被檢索的網頁、郵件、文件或工具回傳內容中。",
      "當模型能呼叫工具、存取私人資料或對外通訊時，風險升高，可能導致資料外洩或未授權的動作。",
      "沒有單一完美的防禦，常見做法是分層：最小權限、對有副作用的動作要求人工確認、隔離不受信任的內容、輸出過濾、監控與紅隊測試。",
    ],
    diagram: [
      "模型讀取外部內容",
      "內容夾帶惡意指令",
      "模型誤把它當成指令",
      "觸發工具呼叫或洩漏資料",
      "以權限限制與人工確認降低影響",
    ],
    code: `# 概念性防護：有副作用的工具需要人工確認
SIDE_EFFECT_TOOLS = {"send_email", "delete_file"}
def execute(call):
    if call.name in SIDE_EFFECT_TOOLS and not user_confirmed(call):
        return "已取消：需要使用者確認"
    return run_tool(call)`,
    caseTitle: "OWASP LLM Top 10 與間接注入研究",
    case: "OWASP 把提示注入列為大型語言模型應用的主要風險之一；Greshake 等人的研究展示了透過被檢索內容進行的間接提示注入。防禦建議需要隨攻擊手法與模型演進持續更新。",
    related: ["function-calling", "react-agent", "model-context-protocol", "rag"],
    sources: [
      ["OWASP Top 10 for LLM Applications", "https://owasp.org/www-project-top-10-for-large-language-model-applications/"],
      ["Not what you've signed up for: Indirect Prompt Injection", "https://arxiv.org/abs/2302.12173"],
    ],
  },
];
