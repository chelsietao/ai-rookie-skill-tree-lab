import type { Concept } from "@/types";

export const deepLearningConcepts: Concept[] = [
  {
    id: "transformer",
    title: "Transformer",
    zh: "Transformer 架構",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "以自注意力（self-attention）為核心、取代循環結構的神經網路架構，是現代大型語言模型的基礎。",
    intuition:
      "像一場會議，每個與會者（token）都可以直接聽取並加權參考所有其他人的發言，而不是只能聽前一位傳話。",
    mechanism: [
      "輸入 token 先轉成向量（embedding），並加入位置資訊，因為自注意力本身不帶順序概念。",
      "每個 Transformer block 通常包含多頭自注意力與前饋網路（FFN），並搭配殘差連接與正規化。",
      "原始論文為編碼器—解碼器結構；許多 LLM 採用 decoder-only，並以 causal mask 限制每個位置只能看到之前的 token。",
      "多個 block 疊加後，輸出層把隱藏狀態轉成詞表上的 logits；Attention 的計算量與序列長度的平方相關，這是長上下文成本的來源。",
    ],
    diagram: [
      "Token embedding + 位置資訊",
      "多頭自注意力（含 mask）",
      "殘差 + 正規化",
      "前饋網路 FFN",
      "重複 N 層後輸出 logits",
    ],
    code: `import torch.nn as nn

layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
)
encoder = nn.TransformerEncoder(layer, num_layers=6)`,
    caseTitle: "Attention Is All You Need",
    case: "2017 年的原始論文提出 Transformer 並在機器翻譯上報告了成果，之後的許多語言模型都以其變體為基礎。各模型在位置編碼、正規化與注意力形式上有不少差異，需看各自的技術報告。",
    related: ["attention", "rope", "layer-normalization", "grouped-query-attention"],
    sources: [
      ["Attention Is All You Need", "https://arxiv.org/abs/1706.03762"],
      ["PyTorch nn.Transformer", "https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html"],
    ],
  },
  {
    id: "rope",
    title: "Rotary Position Embedding",
    zh: "旋轉位置編碼（RoPE）",
    category: "Deep Learning",
    level: "Reference",
    status: "published",
    summary:
      "把位置資訊以旋轉的方式套用到 Query 與 Key 向量上，使注意力分數自然地依賴 token 之間的相對位置。",
    intuition:
      "像給每個詞的向量轉一個角度，角度大小由位置決定；兩個詞比較時，只看角度差，也就是相對距離。",
    mechanism: [
      "把向量的相鄰維度兩兩分成一組，視為二維平面上的點。",
      "依位置 m 對每組套用旋轉角 m·θ，不同組使用不同頻率 θ。",
      "兩個旋轉後向量的內積只與位置差 (m − n) 有關，因此注意力分數具有相對位置的性質。",
      "RoPE 只作用在 Q 與 K，不改變 V；延伸到更長上下文時，常搭配位置插值或頻率縮放等方法，效果需要實測。",
    ],
    diagram: [
      "Q / K 向量",
      "維度兩兩配對",
      "依位置旋轉角度",
      "計算注意力分數（隱含相對位置）",
      "Softmax 與加權 V",
    ],
    code: `import torch

def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    ang = pos[:, None] * inv[None, :]          # [T, d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out`,
    caseTitle: "RoFormer",
    case: "RoFormer 論文提出旋轉位置編碼並在多項 NLP 任務上報告結果；之後有許多開源語言模型採用 RoPE 或其變體。具體採用情況請以各模型的技術文件為準。",
    related: ["transformer", "attention", "kv-cache", "context-window"],
    sources: [
      ["RoFormer paper", "https://arxiv.org/abs/2104.09864"],
      ["Attention Is All You Need", "https://arxiv.org/abs/1706.03762"],
    ],
  },
  {
    id: "layer-normalization",
    title: "LayerNorm and RMSNorm",
    zh: "層正規化與 RMSNorm",
    category: "Deep Learning",
    level: "Reference",
    status: "published",
    summary:
      "對單一樣本的隱藏向量做正規化以穩定訓練；RMSNorm 省略減去平均值的步驟，計算較簡單。",
    intuition:
      "像每一層傳遞訊息前先把音量調整到固定範圍，避免有些層太大聲、有些太小聲，讓後面的層更容易學習。",
    mechanism: [
      "LayerNorm 對每個 token 的特徵維度計算平均值與變異數，標準化後再乘上可學習的縮放並加上偏移。",
      "與 BatchNorm 不同，LayerNorm 不依賴批次內其他樣本，因此適合序列模型與小批次。",
      "RMSNorm 只用均方根（RMS）做縮放，不減去平均值，並保留可學習的縮放參數，運算更少。",
      "Transformer 中正規化的位置（pre-norm 或 post-norm）會影響訓練穩定性，許多大型模型採用 pre-norm。",
    ],
    diagram: [
      "隱藏向量 x",
      "計算統計量（均值 / RMS）",
      "標準化或縮放",
      "可學習縮放（與偏移）",
      "輸出",
    ],
    code: `import torch

def rms_norm(x, weight, eps=1e-6):
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight`,
    caseTitle: "LayerNorm 與 RMSNorm 論文",
    case: "LayerNorm 與 RMSNorm 兩篇論文分別提出兩種方法，後者報告在保持品質的同時降低計算成本。各模型採用哪一種以其技術文件為準。",
    related: ["transformer", "attention", "mixed-precision", "backpropagation"],
    sources: [
      ["Layer Normalization", "https://arxiv.org/abs/1607.06450"],
      ["Root Mean Square Layer Normalization", "https://arxiv.org/abs/1910.07467"],
    ],
  },
  {
    id: "mixture-of-experts",
    title: "Mixture of Experts",
    zh: "專家混合模型",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "模型包含多個「專家」子網路，並由路由器為每個 token 選出少數專家計算，使參數量大但每個 token 的計算量較小。",
    intuition:
      "像一家大型診所有很多專科醫師，但每位病人只會被分診到最相關的一兩位，所以整體知識很多，單次看診成本卻不高。",
    mechanism: [
      "在 Transformer 中，常把部分 FFN 層換成 MoE 層：由多個並列的 FFN（專家）加上一個路由（gating）網路組成。",
      "路由器為每個 token 計算各專家的分數，選出 top-k（常為 1 或 2）個專家，並以權重合併它們的輸出。",
      "為避免少數專家被過度使用，訓練時常加入負載平衡的輔助損失，並設定專家容量上限。",
      "部署時所有專家參數都需要放進記憶體，且需要專家平行（expert parallelism）與 all-to-all 通訊，因此系統複雜度較高。",
    ],
    diagram: [
      "Token 隱藏向量",
      "Router 計算專家分數",
      "選出 top-k 專家",
      "專家各自計算 FFN",
      "加權合併輸出",
    ],
    code: `import torch, torch.nn as nn, torch.nn.functional as F

class TinyMoE(nn.Module):
    def __init__(self, d=64, n_exp=4, k=2):
        super().__init__()
        self.router = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_exp)])
        self.k = k
    def forward(self, x):                      # x: [T, d]
        w, idx = self.router(x).topk(self.k, dim=-1)
        w = F.softmax(w, dim=-1)
        out = torch.zeros_like(x)
        for j in range(self.k):
            for e, expert in enumerate(self.experts):
                m = idx[:, j] == e
                if m.any():
                    out[m] += w[m, j:j+1] * expert(x[m])
        return out`,
    caseTitle: "Switch Transformer 與稀疏門控 MoE",
    case: "Shazeer 等人的稀疏門控 MoE 與 Google 的 Switch Transformer 論文都公開了在語言模型中使用專家路由的設計與實驗結果。MoE 的訓練穩定性與服務成本取決於實作，需要整體系統一起考量。",
    related: ["transformer", "scaling-laws", "tensor-parallelism", "nccl"],
    sources: [
      ["Switch Transformers", "https://arxiv.org/abs/2101.03961"],
      ["Outrageously Large Neural Networks (Sparsely-Gated MoE)", "https://arxiv.org/abs/1701.06538"],
    ],
  },
  {
    id: "backpropagation",
    title: "Backpropagation",
    zh: "反向傳播",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "利用微積分的鏈鎖律，從損失函數往回逐層計算每個參數的梯度，供優化器更新參數。",
    intuition:
      "像成績不理想時，從最終結果往回追查每個環節各自貢獻了多少誤差，再各自微調。",
    mechanism: [
      "前向傳播計算每層輸出並保存必要的中間值，最後得到損失 L。",
      "反向傳播從 ∂L/∂輸出 開始，依鏈鎖律逐層計算 ∂L/∂參數 與要往前傳遞的梯度。",
      "每一層只需知道自己的局部導數與上游傳來的梯度，因此整個網路的梯度可在一次反向掃描中算完。",
      "為了計算梯度必須保留前向的啟動值，這是訓練比推論更吃記憶體的主因之一，也是 gradient checkpointing 的動機。",
    ],
    diagram: [
      "前向傳播並保存中間值",
      "計算損失 L",
      "從輸出層往回傳遞梯度",
      "各層算出參數梯度",
      "優化器更新參數",
    ],
    code: `# 手動示範：y = (w * x - t)^2
w, x, t = 2.0, 3.0, 5.0
y = (w * x - t) ** 2
dy_dw = 2 * (w * x - t) * x    # 鏈鎖律：2(wx-t) * x
print(y, dy_dw)`,
    caseTitle: "Nature 1986 論文與 CS231n 講義",
    case: "Rumelhart、Hinton 與 Williams 在 1986 年的論文使反向傳播廣為人知；史丹佛 CS231n 的課程筆記以計算圖角度清楚說明了推導。這是基礎概念，所有現代深度學習框架都內建其自動化實作。",
    related: ["autograd", "gradient-descent", "computation-graph", "gradient-checkpointing"],
    sources: [
      ["Learning representations by back-propagating errors", "https://www.nature.com/articles/323533a0"],
      ["CS231n: Backpropagation, Intuitions", "https://cs231n.github.io/optimization-2/"],
    ],
  },
  {
    id: "autograd",
    title: "Autograd",
    zh: "自動微分",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "深度學習框架在前向計算時自動記錄運算，並在需要時自動執行反向傳播算出梯度，使用者不必手寫導數。",
    intuition:
      "像一邊做事一邊自動寫下操作日誌；需要倒推責任時，只要照日誌倒著走一遍就好。",
    mechanism: [
      "PyTorch 在張量設定 requires_grad=True 後，對其運算動態建立計算圖，每個結果記錄產生它的 grad_fn。",
      "呼叫 loss.backward() 時，從 loss 出發，沿計算圖反向逐節點呼叫對應的導數函式，並把梯度累加到葉節點的 .grad。",
      "計算圖是動態建立的（define-by-run），每次前向都會產生新圖，因此可以使用 Python 的控制流程。",
      "梯度預設會累加，所以每個訓練步驟前需要清零；推論時用 torch.no_grad() 可以不建立圖、節省記憶體。",
    ],
    diagram: [
      "requires_grad=True 的張量",
      "前向運算並記錄 grad_fn",
      "loss.backward()",
      "沿圖反向計算梯度",
      "梯度累加於 .grad",
    ],
    code: `import torch

w = torch.tensor(2.0, requires_grad=True)
x, t = torch.tensor(3.0), torch.tensor(5.0)
loss = (w * x - t) ** 2
loss.backward()
print(w.grad)        # tensor(6.) = 2*(wx-t)*x`,
    caseTitle: "PyTorch autograd 機制",
    case: "PyTorch 官方文件詳述了 autograd 的運作方式、梯度累加與 no_grad 等用法。這個機制是 torch.compile 與 AOTAutograd 等後續編譯技術的基礎。",
    related: ["backpropagation", "computation-graph", "pytorch", "torch-compile"],
    sources: [
      ["PyTorch autograd mechanics", "https://docs.pytorch.org/docs/stable/notes/autograd.html"],
      ["PyTorch torch.autograd", "https://docs.pytorch.org/docs/stable/autograd.html"],
    ],
  },
  {
    id: "mixed-precision",
    title: "Mixed Precision Training",
    zh: "混合精度訓練",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "在訓練中同時使用 FP16 或 BF16 與 FP32：多數運算用低精度以加速並省記憶體，關鍵部分維持高精度以保持穩定。",
    intuition:
      "像日常記帳用整數元，重要的總帳才用精確到小數的數字：大多數步驟用省力的粗略方式，少數關鍵處才用精細的方式。",
    mechanism: [
      "維護一份 FP32 的主權重（master weights）用於更新，前向與反向則以 FP16 或 BF16 進行，以利用 Tensor Core。",
      "FP16 的數值範圍小，梯度可能下溢，因此常用 loss scaling：先放大損失再反向傳播，更新前把梯度縮回。",
      "BF16 的指數範圍與 FP32 相同，通常不需要 loss scaling，但尾數較少、精度較低。",
      "部分運算（例如 softmax、規約與某些損失函數）常自動維持 FP32；PyTorch 的 autocast 會依規則自動選擇精度。",
    ],
    diagram: [
      "FP32 主權重",
      "轉為 BF16 / FP16 做前向",
      "（FP16）loss scaling",
      "低精度反向傳播",
      "FP32 更新主權重",
    ],
    code: `import torch

model = torch.nn.Linear(1024, 1024).cuda()
opt = torch.optim.AdamW(model.parameters())
x = torch.randn(32, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x).pow(2).mean()
loss.backward()
opt.step(); opt.zero_grad()`,
    caseTitle: "Mixed Precision Training 論文",
    case: "NVIDIA 與百度研究者的論文提出了維護 FP32 主權重加上 loss scaling 的 FP16 訓練方法，並報告了在多種模型上維持精度。PyTorch 的 AMP 文件說明了實務用法。",
    related: ["tensor-cores", "quantization", "gradient-accumulation", "pytorch"],
    sources: [
      ["Mixed Precision Training", "https://arxiv.org/abs/1710.03740"],
      ["PyTorch automatic mixed precision", "https://docs.pytorch.org/docs/stable/amp.html"],
    ],
  },
  {
    id: "tokenization",
    title: "Tokenization",
    zh: "分詞",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "把文字切成模型能處理的 token（子詞單位）並對應到整數 ID，常見方法包含 BPE 與 SentencePiece。",
    intuition:
      "像把句子拆成常見的積木：常用詞是整塊積木，罕見詞就由小積木拼起來，這樣積木種類有限，又能拼出任何句子。",
    mechanism: [
      "字詞層級的詞表會遇到未登錄詞，字元層級的序列又太長；子詞（subword）折衷兩者。",
      "BPE 從單一字元開始，反覆合併語料中最常一起出現的相鄰單位，直到詞表達到目標大小。",
      "SentencePiece 直接把原始文字當成字元序列訓練，不依賴事先的空白切詞，因此對中文、日文等無空白語言也適用。",
      "Token 數直接影響成本、延遲與上下文長度；同一段文字在不同分詞器下的 token 數可能差很多，中文常比英文需要更多 token，需用目標模型的分詞器實際計算。",
    ],
    diagram: [
      "原始文字",
      "正規化",
      "依詞表切成 subword",
      "對應成 token ID",
      "輸入 embedding 層",
    ],
    code: `from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok("Tokenization matters.")["input_ids"]
print(ids, tok.convert_ids_to_tokens(ids))`,
    caseTitle: "BPE 與 SentencePiece",
    case: "Sennrich 等人把 BPE 應用於神經機器翻譯的子詞切分，SentencePiece 論文則提出不依賴預先分詞的實作。Hugging Face 文件整理了常見的分詞演算法。",
    related: ["embeddings", "transformer", "context-window", "latency-throughput"],
    sources: [
      ["Neural Machine Translation of Rare Words with Subword Units", "https://arxiv.org/abs/1508.07909"],
      ["SentencePiece paper", "https://arxiv.org/abs/1808.06226"],
      ["Hugging Face tokenizer summary", "https://huggingface.co/docs/transformers/tokenizer_summary"],
    ],
  },
  {
    id: "embeddings",
    title: "Embeddings",
    zh: "嵌入向量",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "把離散的對象（詞、句子、圖片等）映射成稠密的實數向量，使語意相近的對象在向量空間中距離較近。",
    intuition:
      "像為每個詞在地圖上標一個座標：意思相近的詞住在附近，所以可以用「距離」衡量語意相似度。",
    mechanism: [
      "最基本的 embedding 是一張可學習的查找表：token ID 對應到一個固定維度的向量。",
      "模型訓練時，這些向量會被調整成對任務有用的表示；Word2vec 等方法以「預測上下文」學得詞向量。",
      "句子或文件嵌入通常由 Transformer 編碼後做池化（pooling）得到，並以對比學習等目標訓練，使相似文本的向量接近。",
      "相似度常用餘弦相似度或內積計算；不同模型產生的向量空間不能混用，更換 embedding 模型後需要重新建立索引。",
    ],
    diagram: [
      "文字或 token",
      "Embedding 模型編碼",
      "稠密向量",
      "以餘弦相似度比較",
      "用於檢索、分群或分類",
    ],
    code: `import torch.nn.functional as F
# a, b 為兩個句子的 embedding 向量（形狀 [d]）
sim = F.cosine_similarity(a, b, dim=0)`,
    caseTitle: "Word2vec 與 Sentence-BERT",
    case: "Word2vec 論文展示了詞向量可以捕捉語意與關係，Sentence-BERT 則把 Transformer 用於產生可直接比較的句向量。這是檢索增強生成與語意搜尋的基礎元件。",
    related: ["tokenization", "vector-database", "rag", "transformer"],
    sources: [
      ["Efficient Estimation of Word Representations (word2vec)", "https://arxiv.org/abs/1301.3781"],
      ["Sentence-BERT", "https://arxiv.org/abs/1908.10084"],
    ],
  },
  {
    id: "scaling-laws",
    title: "Scaling Laws",
    zh: "規模法則",
    category: "Deep Learning",
    level: "Reference",
    status: "published",
    summary:
      "描述語言模型的損失如何隨參數量、資料量與算力呈冪律變化的經驗規律，用來規劃訓練資源的分配。",
    intuition:
      "像蓋房子要同時考慮預算、材料與工時：只加大其中一項，效益會遞減；三者搭配得當才划算。",
    mechanism: [
      "Kaplan 等人發現在一定範圍內，測試損失與參數量、資料量、算力大致呈冪律關係。",
      "Chinchilla 研究重新評估後指出，在固定算力下，模型參數與訓練 token 應該以相近比例一起增加，當時許多模型相對於資料量偏大。",
      "實務上用小規模實驗擬合曲線，外推預測大型訓練的損失，並決定模型大小與訓練 token 數。",
      "這些是經驗規律，受資料品質、架構、訓練設定影響；若考慮推論成本，最佳選擇也可能偏向訓練更久的較小模型。",
    ],
    diagram: [
      "小規模實驗",
      "擬合損失 vs 參數／資料／算力",
      "外推預測",
      "決定模型大小與 token 數",
      "正式訓練並驗證",
    ],
    code: `# 概念性冪律：L(N) = (Nc / N) ** alpha
Nc, alpha = 1.0e13, 0.076     # 僅為示意，非任何論文的實際擬合結果
def loss(N): return (Nc / N) ** alpha`,
    caseTitle: "Kaplan 與 Chinchilla",
    case: "OpenAI 的 Kaplan 等人與 DeepMind 的 Hoffmann 等人先後發表規模法則研究，兩者對參數與資料如何分配得到不同結論，常被一起比較。具體係數依資料與設定而異。",
    related: ["pretraining", "mixture-of-experts", "transformer", "latency-throughput"],
    sources: [
      ["Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"],
      ["Training Compute-Optimal Large Language Models", "https://arxiv.org/abs/2203.15556"],
    ],
  },
  {
    id: "pytorch",
    title: "PyTorch",
    zh: "PyTorch 深度學習框架",
    category: "Deep Learning",
    level: "Core focus",
    status: "published",
    summary:
      "以動態計算圖與 Python 優先為特色的開源深度學習框架，提供張量運算、自動微分、GPU 加速與分散式訓練。",
    intuition:
      "像一個寫起來就像一般 Python 的深度學習工具箱：想怎麼寫就怎麼寫，需要時再用編譯器把它變快。",
    mechanism: [
      "核心資料結構是 Tensor，可放在 CPU 或 GPU，並支援自動微分（autograd）。",
      "以 nn.Module 組織模型，optimizer 負責更新參數，DataLoader 負責資料載入。",
      "Eager mode 逐行執行、容易除錯；torch.compile 可把模型編譯成最佳化後的程式碼以提升效能。",
      "torch.distributed 提供 DDP、FSDP 等分散式訓練工具，後端常使用 NCCL。",
    ],
    diagram: [
      "Tensor 與 nn.Module",
      "前向計算（eager）",
      "autograd 反向傳播",
      "optimizer 更新",
      "（可選）torch.compile / 分散式",
    ],
    code: `import torch, torch.nn as nn

model = nn.Linear(10, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x, y = torch.randn(32, 10), torch.randn(32, 1)
loss = nn.functional.mse_loss(model(x), y)
loss.backward(); opt.step(); opt.zero_grad()`,
    caseTitle: "PyTorch 論文與官方文件",
    case: "PyTorch 論文說明了其以命令式、Pythonic 風格兼顧效能的設計理念；官方文件提供完整的 API 與教學。許多推論與訓練系統（例如 vLLM）都建立在 PyTorch 之上。",
    related: ["autograd", "torch-compile", "ddp", "mixed-precision"],
    sources: [
      ["PyTorch paper", "https://arxiv.org/abs/1912.01703"],
      ["PyTorch documentation", "https://docs.pytorch.org/docs/stable/index.html"],
    ],
  },
];
