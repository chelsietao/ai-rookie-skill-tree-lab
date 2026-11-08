import type { Concept } from "@/types";

export const mlProgrammingConcepts: Concept[] = [
  {
    id: "ml-evaluation",
    title: "ML Evaluation",
    zh: "機器學習評估",
    category: "Machine Learning",
    level: "Core focus",
    status: "published",
    summary:
      "以合適的資料切分、指標、基準線與錯誤分析衡量模型能力，避免被偏誤的評估誤導。",
    intuition:
      "像考試要出沒看過的題目：用訓練時看過的題目考試，分數再高也不代表真的學會了。",
    mechanism: [
      "把資料切成訓練、驗證與測試集：訓練集用來學習，驗證集用來調整與選模型，測試集只在最後用來估計泛化表現。",
      "選擇與目標一致的指標：分類看精確率、召回率、F1 或 AUC，迴歸看 MAE 或 RMSE，排序與檢索有各自的指標。",
      "建立簡單的基準線（如多數類別或線性模型）作為比較，確認複雜模型是否真的帶來改善。",
      "進行錯誤分析：觀察哪些類型的樣本出錯、不同子群體表現是否不均，並留意資料洩漏與分佈與線上不一致的問題。",
    ],
    diagram: [
      "切分訓練／驗證／測試",
      "訓練並用驗證集調整",
      "鎖定模型後在測試集評估",
      "與基準線比較",
      "錯誤分析並迭代",
    ],
    code: `from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te)))`,
    caseTitle: "scikit-learn 模型評估指南",
    case: "scikit-learn 文件整理了常見的評估指標與使用方式，並說明了交叉驗證與評分函式。選擇指標時應從業務目標出發，而非只看單一數字。",
    related: ["cross-validation", "classification-metrics", "overfitting-regularization", "llm-as-judge"],
    sources: [
      ["scikit-learn model evaluation", "https://scikit-learn.org/stable/modules/model_evaluation.html"],
      ["scikit-learn cross-validation", "https://scikit-learn.org/stable/modules/cross_validation.html"],
    ],
  },
  {
    id: "gradient-descent",
    title: "Gradient Descent",
    zh: "梯度下降",
    category: "Machine Learning",
    level: "Core focus",
    status: "published",
    summary:
      "沿著損失函數梯度的反方向逐步更新參數，使損失逐漸下降，是訓練神經網路的基本優化方法。",
    intuition:
      "像在濃霧中下山：看不到山谷，只能感受腳下哪個方向最陡，朝下坡走一小步，再重新判斷。",
    mechanism: [
      "計算損失 L(θ) 對參數 θ 的梯度 ∇L，它指向損失上升最快的方向。",
      "以 θ ← θ − η·∇L 更新參數，其中 η 為學習率；太大會發散，太小則收斂緩慢。",
      "實務上用小批次估計梯度（mini-batch SGD），計算便宜並帶有雜訊，有助於逃離某些不良區域。",
      "常見改良包含動量（momentum）與自適應方法（Adam、AdamW），它們會依歷史梯度調整每個參數的步長。",
    ],
    diagram: [
      "目前參數 θ",
      "計算損失與梯度",
      "沿負梯度更新",
      "新的參數與較低損失",
      "重複直到收斂",
    ],
    code: `# 對 f(x) = (x - 3)^2 做梯度下降
x, lr = 0.0, 0.1
for _ in range(50):
    grad = 2 * (x - 3)
    x -= lr * grad
print(x)   # 接近 3`,
    caseTitle: "Adam 優化器",
    case: "Kingma 與 Ba 的 Adam 論文提出了結合動量與自適應學習率的優化器，現今廣泛使用於深度學習；其權重衰減的變體 AdamW 則由後續研究提出。優化器的選擇與超參數須依任務調整。",
    related: ["backpropagation", "learning-rate-schedule", "overfitting-regularization", "autograd"],
    sources: [
      ["Adam: A Method for Stochastic Optimization", "https://arxiv.org/abs/1412.6980"],
      ["An overview of gradient descent optimization algorithms", "https://arxiv.org/abs/1609.04747"],
    ],
  },
  {
    id: "overfitting-regularization",
    title: "Overfitting and Regularization",
    zh: "過擬合與正則化",
    category: "Machine Learning",
    level: "Core focus",
    status: "published",
    summary:
      "過擬合指模型過度記住訓練資料的細節與雜訊，在新資料上表現變差；正則化是降低過擬合的各種方法。",
    intuition:
      "像學生把考古題的答案背下來，遇到沒見過的題目就不會；正則化就是逼他理解原理，而不是死記。",
    mechanism: [
      "症狀：訓練損失持續下降，但驗證損失在某個時間點之後開始上升。",
      "常見原因：模型容量相對於資料量太大、訓練太久、資料量不足或標籤雜訊。",
      "方法包含權重衰減（L2 正則化）、Dropout、資料增強、早停（early stopping）、取得更多資料或簡化模型。",
      "需要用驗證集監控泛化表現；過度正則化會造成欠擬合，所以強度需要調整，並與基準線比較。",
    ],
    diagram: [
      "訓練並監控訓練／驗證損失",
      "驗證損失開始上升",
      "套用正則化或早停",
      "重新訓練並評估",
      "選擇泛化最好的模型",
    ],
    code: `import torch

model = torch.nn.Sequential(
    torch.nn.Linear(100, 64), torch.nn.ReLU(),
    torch.nn.Dropout(p=0.3), torch.nn.Linear(64, 2),
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`,
    caseTitle: "Dropout 論文",
    case: "Srivastava 等人的論文提出 Dropout，並在多個任務上報告了降低過擬合的效果。大型語言模型因資料量極大，常用的正則化組合與小模型不同。",
    related: ["ml-evaluation", "cross-validation", "gradient-descent", "classification-metrics"],
    sources: [
      ["Dropout: A Simple Way to Prevent Overfitting", "https://jmlr.org/papers/v15/srivastava14a.html"],
      ["scikit-learn: underfitting vs overfitting", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html"],
    ],
  },
  {
    id: "cross-validation",
    title: "Cross-Validation",
    zh: "交叉驗證",
    category: "Machine Learning",
    level: "Reference",
    status: "published",
    summary:
      "把資料分成多折，輪流用其中一折驗證、其餘訓練，以更穩定地估計模型的泛化表現。",
    intuition:
      "像讓同一位學生輪流考五份不同的試卷再平均，比只考一次更能反映真實實力。",
    mechanism: [
      "K-fold：把資料分成 K 份，每次用 K−1 份訓練、1 份驗證，共進行 K 輪，最後平均各輪分數並看其變異。",
      "分類問題常用分層（stratified）切分，使每折的類別比例一致；時間序列要用依時間順序的切法，避免用未來資料預測過去。",
      "若同一個使用者或群組的資料可能出現在訓練與驗證兩邊，要用 group K-fold 防止洩漏。",
      "超參數搜尋與最終評估若用同一份資料會高估表現，需使用巢狀交叉驗證或另外保留測試集。",
    ],
    diagram: [
      "資料分成 K 折",
      "輪流指定一折為驗證集",
      "訓練並計算分數",
      "平均 K 個分數與變異",
      "決定模型與超參數",
    ],
    code: `from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
print(scores.mean(), scores.std())`,
    caseTitle: "scikit-learn 的交叉驗證工具",
    case: "scikit-learn 文件說明了 K-fold、分層、群組與時間序列等切分策略及其適用情境。對深度學習大模型，因訓練成本高，常以單一驗證集代替完整交叉驗證。",
    related: ["ml-evaluation", "overfitting-regularization", "classification-metrics", "experiment-tracking"],
    sources: [
      ["scikit-learn cross-validation", "https://scikit-learn.org/stable/modules/cross_validation.html"],
      ["scikit-learn model selection", "https://scikit-learn.org/stable/model_selection.html"],
    ],
  },
  {
    id: "classification-metrics",
    title: "Classification Metrics",
    zh: "分類評估指標",
    category: "Machine Learning",
    level: "Core focus",
    status: "published",
    summary:
      "衡量分類模型表現的指標，包含準確率、精確率、召回率、F1 與 ROC AUC，各自強調不同類型的錯誤。",
    intuition:
      "像醫療檢查：「有病卻沒檢出」和「沒病卻誤報」是兩種不同代價的錯誤，只看總體答對率會看不出這個差別。",
    mechanism: [
      "混淆矩陣把預測分成 TP、FP、TN、FN 四類，是多數指標的基礎。",
      "精確率 = TP/(TP+FP)，強調被判為正的有多少是真的正；召回率 = TP/(TP+FN)，強調真的正的有多少被找出來；F1 是兩者的調和平均。",
      "類別不平衡時，準確率可能很高卻沒用（全猜多數類別）；此時應看精確率、召回率、PR 曲線或 F1。",
      "ROC AUC 衡量不同門檻下區分正負樣本的能力；選擇門檻時要依錯誤的實際代價決定，而非固定使用 0.5。",
    ],
    diagram: [
      "模型輸出機率或類別",
      "依門檻得到預測",
      "混淆矩陣",
      "計算精確率／召回率／F1",
      "依錯誤代價選擇門檻",
    ],
    code: `from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(roc_auc_score(y_true, y_prob))`,
    caseTitle: "scikit-learn 指標與 PR 曲線研究",
    case: "scikit-learn 文件定義了常見分類指標；Davis 與 Goadrich 的研究比較了 ROC 與 Precision-Recall 曲線的關係，指出在類別不平衡時 PR 曲線可能更具資訊。",
    related: ["ml-evaluation", "cross-validation", "overfitting-regularization", "llm-as-judge"],
    sources: [
      ["scikit-learn classification metrics", "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics"],
      ["The relationship between Precision-Recall and ROC curves", "https://dl.acm.org/doi/10.1145/1143844.1143874"],
    ],
  },
  {
    id: "python",
    title: "Python",
    zh: "Python 程式語言",
    category: "Programming",
    level: "Core focus",
    status: "published",
    summary:
      "AI 原型開發、資料處理與框架 API 最常用的語言，效能關鍵的部分通常由 C/C++/CUDA 撰寫的函式庫負責。",
    intuition:
      "像一位擅長溝通協調的專案經理：自己不一定做最重的體力活，但能把各路專家（高效能函式庫）組織起來完成工作。",
    mechanism: [
      "Python 是直譯式、動態型別的語言，開發速度快，生態系包含 NumPy、PyTorch、pandas 與 scikit-learn 等。",
      "效能關鍵的運算（矩陣乘法、卷積）由底層以 C/C++/CUDA 實作的函式庫執行，Python 負責描述流程。",
      "迴圈密集的純 Python 程式很慢，應改用向量化運算、編譯工具或把熱點改寫成原生擴充。",
      "工程實踐包含虛擬環境與固定相依版本、型別標註、測試、日誌與打包，這些對可重現與可維護至關重要。",
    ],
    diagram: [
      "Python 程式描述流程",
      "呼叫框架 API",
      "C++ / CUDA 後端執行",
      "回傳結果",
      "（熱點）向量化或原生擴充",
    ],
    code: `import numpy as np

x = np.random.rand(1_000_000)
# 向量化：由底層 C 實作完成，遠快於 Python 迴圈
y = (x * 2 + 1).sum()`,
    caseTitle: "Python 在 AI 生態的角色",
    case: "PyTorch 與 NumPy 等函式庫以 Python 為主要介面、底層以原生碼實作；Python 官方文件提供語言與標準函式庫的說明。學習重點是理解何時語言本身是瓶頸、何時不是。",
    related: ["numpy-broadcasting", "python-gil", "pytorch", "pybind11"],
    sources: [
      ["Python documentation", "https://docs.python.org/3/"],
      ["The Python Tutorial", "https://docs.python.org/3/tutorial/index.html"],
    ],
  },
  {
    id: "numpy-broadcasting",
    title: "NumPy Vectorization and Broadcasting",
    zh: "NumPy 向量化與廣播",
    category: "Programming",
    level: "Core focus",
    status: "published",
    summary:
      "以整個陣列為單位的運算取代 Python 迴圈（向量化），並以廣播規則讓不同形狀的陣列能一起運算。",
    intuition:
      "像一次下指令「全班每個人加五分」，而不是逐一叫名字加分；廣播則像把一張單行的小抄自動複印成與全班一樣大。",
    mechanism: [
      "NumPy 陣列在連續記憶體中儲存同型別資料，運算由 C 實作的迴圈執行，因此比 Python 迴圈快得多。",
      "廣播規則：從最後一個維度往前比對，兩者維度相同或其中一個為 1（或不存在）才相容，維度為 1 的會被虛擬地擴展，不會真正複製資料。",
      "形狀不相容時會報錯（例如 (3,) 與 (4,)）；常見錯誤包含意外廣播成大矩陣，造成記憶體暴增或結果錯誤。",
      "PyTorch 與 JAX 的張量使用相同的廣播規則，熟悉它對讀懂模型程式碼非常重要。",
    ],
    diagram: [
      "陣列 A 形狀 (3, 1)",
      "陣列 B 形狀 (1, 4)",
      "從最後維度比對並擴展維度 1",
      "結果形狀 (3, 4)",
      "逐元素運算",
    ],
    code: `import numpy as np

a = np.arange(3).reshape(3, 1)   # (3, 1)
b = np.arange(4).reshape(1, 4)   # (1, 4)
print((a + b).shape)             # (3, 4)`,
    caseTitle: "NumPy 官方文件",
    case: "NumPy 官方文件以圖示說明廣播規則與其限制。深度學習框架沿用相同概念，所以這是閱讀與除錯張量程式的基本功。",
    related: ["python", "pytorch", "attention", "embeddings"],
    sources: [
      ["NumPy broadcasting", "https://numpy.org/doc/stable/user/basics.broadcasting.html"],
      ["PyTorch broadcasting semantics", "https://docs.pytorch.org/docs/stable/notes/broadcasting.html"],
    ],
  },
  {
    id: "python-gil",
    title: "Python GIL",
    zh: "Python 全域直譯器鎖",
    category: "Programming",
    level: "Reference",
    status: "published",
    summary:
      "CPython 傳統上以全域直譯器鎖（GIL）確保同一時刻只有一個執行緒執行 Python 位元碼，影響多執行緒 CPU 密集程式的並行度。",
    intuition:
      "像一間廚房只有一支主廚專用的刀，大家可以同時準備食材，但真正切菜的動作一次只有一個人能用刀。",
    mechanism: [
      "GIL 保護 CPython 直譯器內部狀態（例如引用計數）的一致性，同一時刻只有持有 GIL 的執行緒能執行 Python 位元碼。",
      "I/O 等待與許多由 C 實作的函式庫（如 NumPy 的部分運算）會在執行時釋放 GIL，因此多執行緒對 I/O 密集任務仍然有效。",
      "CPU 密集的純 Python 程式用多執行緒無法利用多核心，常見替代方案是多行程（multiprocessing）、把熱點移到原生擴充或使用非同步 I/O。",
      "近期的 CPython 版本提供了可選的 free-threaded（無 GIL）建置，但生態系相容性仍在演進，採用前需檢查所用套件的支援狀況。",
    ],
    diagram: [
      "多個執行緒",
      "競爭 GIL",
      "持有者執行 Python 位元碼",
      "I/O 或原生碼釋放 GIL",
      "其他執行緒接手",
    ],
    code: `from concurrent.futures import ProcessPoolExecutor

def heavy(n):
    return sum(i * i for i in range(n))

with ProcessPoolExecutor() as pool:     # 以多行程繞過 GIL 的限制
    print(list(pool.map(heavy, [10**6] * 4)))`,
    caseTitle: "資料載入與多行程",
    case: "PyTorch 的 DataLoader 以多個 worker 行程讀取與前處理資料，部分原因就是避免 GIL 限制 CPU 密集的前處理。Python 官方文件說明了 GIL 與可選的無 GIL 建置。",
    related: ["python", "numpy-broadcasting", "pybind11", "pytorch"],
    sources: [
      ["Python glossary: global interpreter lock", "https://docs.python.org/3/glossary.html#term-global-interpreter-lock"],
      ["Python free-threading HOWTO", "https://docs.python.org/3/howto/free-threading-python.html"],
    ],
  },
  {
    id: "pybind11",
    title: "pybind11",
    zh: "C++ 與 Python 綁定",
    category: "Programming",
    level: "Reference",
    status: "published",
    summary:
      "輕量的 C++ 函式庫，用來把 C++ 的函式與類別暴露給 Python，是撰寫高效能 Python 擴充模組的常用工具。",
    intuition:
      "像在 Python 與 C++ 之間蓋一座橋：效能關鍵的部分用 C++ 寫，Python 端只要像呼叫一般函式一樣使用它。",
    mechanism: [
      "在 C++ 中以 PYBIND11_MODULE 巨集定義模組，並用 m.def 綁定函式、用 py::class_ 綁定類別。",
      "pybind11 處理 C++ 與 Python 物件之間的型別轉換，並支援與 NumPy 陣列的緩衝區（buffer protocol）互通，避免不必要的資料複製。",
      "透過 CMake 或 setuptools 編譯成共享函式庫（.so／.pyd），然後在 Python 中 import。",
      "要注意物件生命週期、GIL 的釋放（長時間運算前可釋放以允許其他執行緒執行）與例外轉換；PyTorch 的 C++ 擴充也有自己的建置工具。",
    ],
    diagram: [
      "撰寫 C++ 函式",
      "以 pybind11 宣告模組與綁定",
      "編譯成共享函式庫",
      "Python import",
      "呼叫 C++ 實作",
    ],
    code: `// example.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(example, m) {
    m.def("add", &add, "Add two integers");
}`,
    caseTitle: "PyTorch 與原生擴充",
    case: "pybind11 官方文件說明了綁定函式、類別與 NumPy 陣列的方式；PyTorch 的 Python 綁定也使用 pybind11。這是理解「Python 介面、C++ 核心」架構的好入口。",
    related: ["cpp", "python", "python-gil", "cuda"],
    sources: [
      ["pybind11 documentation", "https://pybind11.readthedocs.io/en/stable/"],
      ["pybind11 repository", "https://github.com/pybind/pybind11"],
    ],
  },
  {
    id: "cpp",
    title: "Modern C++",
    zh: "現代 C++",
    category: "Programming",
    level: "Core focus",
    status: "published",
    summary:
      "高效能系統、推論引擎、深度學習框架核心與 GPU host 端開發的重要語言，現代 C++ 強調 RAII、智慧指標與移動語意。",
    intuition:
      "像可以直接操作引擎零件的賽車手冊：能榨出最高的效能，但也要自己負責每個零件的安裝與回收。",
    mechanism: [
      "RAII（資源取得即初始化）把資源的生命週期綁在物件上：建構時取得、解構時釋放，離開作用域自動清理，減少洩漏。",
      "std::unique_ptr 與 std::shared_ptr 管理動態記憶體的擁有權；移動語意（std::move）讓資源能轉移而非複製，降低大型物件的成本。",
      "模板（templates）支援泛型與編譯期最佳化，許多張量與 GPU 函式庫大量使用；標準函式庫提供容器、演算法與並行工具。",
      "在 AI 系統中常見的用途包含推論引擎、自訂運算子與 CUDA host 程式；需要特別注意記憶體對齊、執行緒安全與未定義行為。",
    ],
    diagram: [
      "建構物件並取得資源",
      "使用資源",
      "離開作用域自動解構",
      "釋放資源",
      "（需要時）移動所有權",
    ],
    code: `#include <memory>
#include <vector>

struct Buffer {
    std::vector<float> data;
    explicit Buffer(size_t n) : data(n) {}
};

int main() {
    auto p = std::make_unique<Buffer>(1024);   // 離開作用域自動釋放
    auto q = std::move(p);                     // 所有權轉移給 q
}`,
    caseTitle: "ISO C++ 與 C++ Core Guidelines",
    case: "ISO C++ 基金會網站提供標準與學習資源，C++ Core Guidelines 則由社群與標準委員會成員維護，整理了現代 C++ 的寫法建議。AI 工程師多半不需精通全部細節，但理解資源管理與效能模型很有幫助。",
    related: ["cuda", "pybind11", "python", "model-serving"],
    sources: [
      ["Standard C++ Foundation", "https://isocpp.org/"],
      ["C++ Core Guidelines", "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines"],
    ],
  },
];
