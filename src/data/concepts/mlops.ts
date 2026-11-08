import type { Concept } from "@/types";

export const mlopsConcepts: Concept[] = [
  {
    id: "model-serving",
    title: "Model Serving",
    zh: "模型線上服務",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "把訓練好的模型部署成可靠、可擴展、可監控的服務，處理併發請求、批次、版本、延遲與失敗。",
    intuition:
      "像把一位專家安排進客服中心：除了專業本身，還要有接線、排隊、備援與品質監控，才能穩定服務大量來電。",
    mechanism: [
      "服務層接收請求（HTTP 或 gRPC），進行驗證、前處理與排隊，並依動態批次（dynamic batching）把請求合併後送進推論引擎。",
      "推論引擎載入模型（常為最佳化後的格式，如 TensorRT engine 或 ONNX），在 GPU 或 CPU 上執行，並回傳後處理的結果。",
      "營運面需要版本管理與灰度發布、自動擴縮容、健康檢查、逾時與重試，並以延遲分位數（p50、p95、p99）與吞吐量監控服務水準。",
      "不同模型的特性不同：傳統模型重視批次與延遲；LLM 還需處理串流輸出、KV Cache 與 continuous batching。",
    ],
    diagram: [
      "客戶端請求",
      "負載平衡與認證",
      "排隊與動態批次",
      "推論引擎（GPU / CPU）",
      "回應與監控指標",
    ],
    code: `# FastAPI 最小範例（示意；生產環境需加入批次、逾時與監控）
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(x: list[float]):
    return {"y": model(x)}`,
    caseTitle: "Triton 與 KServe",
    case: "NVIDIA Triton Inference Server 與 KServe 都是開源的模型服務方案，文件中說明了動態批次、多模型管理與在 Kubernetes 上部署的做法。具體選型取決於模型類型、硬體與團隊既有平台。",
    related: ["latency-throughput", "continuous-batching", "triton-inference-server", "kubernetes"],
    sources: [
      ["NVIDIA Triton Inference Server", "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/"],
      ["KServe documentation", "https://kserve.github.io/website/"],
    ],
  },
  {
    id: "experiment-tracking",
    title: "Experiment Tracking",
    zh: "實驗追蹤",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "系統化記錄每次訓練的程式版本、資料版本、超參數、指標與產物，使實驗可比較、可重現。",
    intuition:
      "像實驗室筆記本：每次實驗用了什麼配方、得到什麼結果都寫下來，之後才能知道哪一次最好、為什麼好。",
    mechanism: [
      "每次執行（run）記錄參數（超參數、設定）、指標（loss、accuracy 隨步數變化）、產物（模型檔、圖表）與環境（程式碼 commit、套件版本）。",
      "追蹤伺服器或服務儲存這些資料，並提供介面比較不同 run、依指標排序與篩選。",
      "與資料版本控制、隨機種子與設定檔管理搭配，才能真正重現結果；只記錄指標而不記錄資料與程式版本，容易無法重現。",
      "常見工具包含 MLflow、TensorBoard 與各類託管服務，選擇時要考慮自架或託管、權限管理與整合成本。",
    ],
    diagram: [
      "開始 run",
      "記錄參數與環境",
      "訓練並記錄指標與產物",
      "結束 run",
      "比較 run 並挑選最佳模型",
    ],
    code: `import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 3e-4)
    for step in range(100):
        mlflow.log_metric("loss", 1.0 / (step + 1), step=step)`,
    caseTitle: "MLflow Tracking",
    case: "MLflow 官方文件說明了以 run 為單位記錄參數、指標與產物的 Tracking 功能。其他工具有各自的模型與介面，概念相近。",
    related: ["model-registry", "data-versioning", "ml-evaluation", "workflow-orchestration"],
    sources: [
      ["MLflow documentation", "https://mlflow.org/docs/latest/"],
      ["TensorBoard", "https://www.tensorflow.org/tensorboard"],
    ],
  },
  {
    id: "model-registry",
    title: "Model Registry",
    zh: "模型登錄庫",
    category: "Data & MLOps",
    level: "Reference",
    status: "published",
    summary:
      "集中管理模型版本、中繼資料與生命週期階段（如 staging、production）的儲存庫，連結訓練與部署。",
    intuition:
      "像藥品的倉儲與出貨系統：每一批都有編號、檢驗紀錄與狀態，出貨前可以確認「現在上架的是哪一個版本」。",
    mechanism: [
      "訓練完成後，把模型產物與其來源（實驗 run、資料版本、程式碼 commit）一起註冊為新版本。",
      "為版本附上標籤或別名（例如 candidate、champion）與審核紀錄，部署流程依別名取得要上線的版本。",
      "回滾時只需要把別名指回先前的版本，不必重新訓練或重新打包。",
      "權限與稽核很重要：誰可以註冊、核准、部署，都應留下紀錄；公開託管的模型庫（如 Hugging Face Hub）還要留意授權條款與來源可信度。",
    ],
    diagram: [
      "訓練產出模型",
      "註冊新版本（含來源資訊）",
      "驗證與核准",
      "指定別名／階段",
      "部署流程讀取並上線",
    ],
    code: `import mlflow

# 註冊模型版本（run_id 為某次實驗的 run；以官方文件為準）
mlflow.register_model(f"runs:/{run_id}/model", "fraud-detector")`,
    caseTitle: "MLflow Model Registry 與 Hugging Face Hub",
    case: "MLflow 提供模型登錄功能管理版本與別名；Hugging Face Hub 則是公開的模型與資料集託管平台，文件說明了版本與模型卡的使用。兩者定位不同，適用情境依團隊需求而異。",
    related: ["experiment-tracking", "model-serving", "data-versioning", "distributed-checkpointing"],
    sources: [
      ["MLflow documentation", "https://mlflow.org/docs/latest/"],
      ["Hugging Face Hub documentation", "https://huggingface.co/docs/hub/index"],
    ],
  },
  {
    id: "feature-store",
    title: "Feature Store",
    zh: "特徵儲存庫",
    category: "Data & MLOps",
    level: "Reference",
    status: "published",
    summary:
      "集中定義、計算與提供機器學習特徵的系統，確保訓練與線上推論使用一致的特徵邏輯。",
    intuition:
      "像共用的食材準備區：訓練時與正式出餐時都從同一個地方取料、用同一種切法，避免兩邊的料不一樣。",
    mechanism: [
      "以統一的定義描述特徵（來源、轉換、實體與時間戳），讓不同團隊與模型可以重用。",
      "離線儲存（資料倉儲或資料湖）支援大規模訓練資料集的產生，並以時間點正確（point-in-time correct）的方式取得歷史特徵，避免資料洩漏。",
      "線上儲存（低延遲的鍵值資料庫）提供推論時即時查詢最新特徵；兩邊由同一份定義同步。",
      "主要價值在於避免訓練與服務偏差（training-serving skew）並促進重用；對小型專案可能增加不必要的複雜度。",
    ],
    diagram: [
      "原始資料來源",
      "特徵定義與轉換",
      "離線儲存（訓練）",
      "線上儲存（推論）",
      "模型取得一致的特徵",
    ],
    code: `# Feast 取得線上特徵（示意；以官方文件為準）
features = store.get_online_features(
    features=["driver_stats:avg_rating"],
    entity_rows=[{"driver_id": 1001}],
).to_dict()`,
    caseTitle: "Feast 與技術債論文",
    case: "Feast 是開源的特徵儲存庫，文件說明了離線與線上儲存的設計；Google 研究者的「Hidden Technical Debt in ML Systems」論文討論了機器學習系統中資料與特徵相依性造成的維護成本。",
    related: ["data-drift", "data-versioning", "model-serving", "workflow-orchestration"],
    sources: [
      ["Feast documentation", "https://docs.feast.dev/"],
      ["Hidden Technical Debt in Machine Learning Systems", "https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html"],
    ],
  },
  {
    id: "data-drift",
    title: "Data Drift and Concept Drift",
    zh: "資料漂移與概念漂移",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "上線後的輸入資料分佈（資料漂移）或輸入與標籤的關係（概念漂移）隨時間改變，使模型表現逐漸下降。",
    intuition:
      "像用去年的地圖導航：城市蓋了新路、改了單行道，地圖沒更新，導航就開始出錯。",
    mechanism: [
      "資料漂移：特徵的分佈與訓練時不同，例如使用者族群或感測器校準改變。",
      "概念漂移：同樣的輸入，正確的輸出改變了，例如詐欺手法演變使舊特徵不再有效。",
      "偵測方式包含比較線上與參考資料的分佈（統計檢定、PSI、KL 散度等），並監控預測分佈與在標籤延遲到達後的實際指標。",
      "發現漂移後的回應包含檢查資料管線是否出錯、重新訓練或調整模型，並設定警報門檻；統計上顯著的漂移不一定影響模型表現，需要結合業務指標判斷。",
    ],
    diagram: [
      "保存訓練期的參考分佈",
      "線上資料持續蒐集",
      "比較分佈與預測指標",
      "超過門檻觸發警報",
      "調查並重新訓練或修正",
    ],
    code: `from scipy.stats import ks_2samp

# ref 為訓練期特徵值，cur 為近期線上特徵值
stat, p = ks_2samp(ref, cur)
drifted = p < 0.01     # 僅為示意；實務需考慮樣本數與業務影響`,
    caseTitle: "概念漂移與資料集偏移研究",
    case: "Gama 等人整理了概念漂移適應的研究，Rabanser 等人的論文實證比較了偵測資料集偏移的方法。開源工具如 Evidently 也提供漂移報告，但門檻與處置仍需依情境設計。",
    related: ["ml-observability", "ml-evaluation", "feature-store", "model-serving"],
    sources: [
      ["A survey on concept drift adaptation", "https://dl.acm.org/doi/10.1145/2523813"],
      ["Failing Loudly: Detecting Dataset Shift", "https://arxiv.org/abs/1810.11953"],
    ],
  },
  {
    id: "data-versioning",
    title: "Data Versioning",
    zh: "資料版本控制",
    category: "Data & MLOps",
    level: "Reference",
    status: "published",
    summary:
      "像程式碼一樣為資料集與中間產物建立版本，使每次訓練使用的資料都可追溯與重現。",
    intuition:
      "像為每份報告的資料表存檔並編號：日後想知道某個結果是用哪一版資料算出來的，一查就知道。",
    mechanism: [
      "大型檔案不適合直接放進 Git，工具通常把資料存於物件儲存，Git 只保存指向資料內容雜湊的小型中繼檔。",
      "將資料版本與程式碼 commit、參數與管線定義綁在一起，就能重現某次實驗所用的完整輸入。",
      "資料管線（pipeline）定義產生資料與模型的階段與相依性，只重跑受影響的階段。",
      "需要考慮儲存成本、敏感資料的存取控制與保留政策；版本控制不等於資料品質驗證，兩者應搭配。",
    ],
    diagram: [
      "原始資料集",
      "計算內容雜湊並存入物件儲存",
      "Git 記錄中繼檔",
      "與程式碼版本綁定",
      "任一版本可還原",
    ],
    code: `# DVC 基本流程（命令列示意）
# dvc init
# dvc add data/train.csv
# git add data/train.csv.dvc .gitignore && git commit -m "add dataset v1"
# dvc push`,
    caseTitle: "DVC 與 lakeFS",
    case: "DVC 與 lakeFS 都是開源的資料版本控制工具，前者偏向與 Git 工作流整合，後者提供類似 Git 分支的資料湖版本管理。兩者的模型與適用規模不同。",
    related: ["experiment-tracking", "feature-store", "model-registry", "workflow-orchestration"],
    sources: [
      ["DVC documentation", "https://dvc.org/doc"],
      ["lakeFS documentation", "https://docs.lakefs.io/"],
    ],
  },
  {
    id: "workflow-orchestration",
    title: "Workflow Orchestration",
    zh: "工作流程編排",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "以有向無環圖（DAG）定義資料處理、訓練與評估等任務及其相依關係，由編排系統排程、重試並監控。",
    intuition:
      "像工廠的生產排程表：知道哪一道工序要等哪一道完成，並在某一站出問題時自動重做或通知。",
    mechanism: [
      "把流程拆成任務（task），並定義任務之間的相依關係，形成 DAG。",
      "編排系統依排程或事件觸發，按相依順序在工作節點上執行任務，管理重試、逾時與並行度。",
      "任務之間以中間產物（資料表、檔案、參數）傳遞，良好的設計要讓任務具冪等性，重跑不會產生重複或不一致的結果。",
      "ML 流程常包含資料驗證、特徵、訓練、評估與部署等階段；選擇工具時要考慮與運算平台（如 Kubernetes）的整合與團隊技能。",
    ],
    diagram: [
      "定義任務與相依 DAG",
      "排程或事件觸發",
      "依序／並行執行任務",
      "失敗重試與警報",
      "產物與紀錄保存",
    ],
    code: `from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2026, 1, 1), schedule="@daily", catchup=False)
def train_pipeline():
    @task
    def prepare(): ...
    @task
    def train(data): ...
    train(prepare())

train_pipeline()`,
    caseTitle: "Airflow 與 Kubeflow Pipelines",
    case: "Apache Airflow 是常見的通用工作流程編排工具，Kubeflow Pipelines 則針對在 Kubernetes 上的機器學習流程。兩者的設計目標與使用體驗不同。",
    related: ["experiment-tracking", "data-versioning", "kubernetes", "containerization"],
    sources: [
      ["Apache Airflow documentation", "https://airflow.apache.org/docs/"],
      ["Kubeflow Pipelines", "https://www.kubeflow.org/docs/components/pipelines/"],
    ],
  },
  {
    id: "containerization",
    title: "Containerization",
    zh: "容器化",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "把應用程式與其相依套件、設定打包成可在不同環境一致執行的容器映像檔，讓部署與重現更可靠。",
    intuition:
      "像把整個廚房的設備、食材與食譜裝進標準貨櫃，不管運到哪裡，打開就能用同樣的方式煮出同樣的菜。",
    mechanism: [
      "以 Dockerfile 描述映像檔的建置步驟：基底映像、安裝相依套件、複製程式碼與啟動命令。",
      "容器共用主機核心，但透過命名空間與 cgroups 隔離行程、檔案系統與資源，比虛擬機更輕量。",
      "使用 GPU 時，需要主機安裝驅動並使用 NVIDIA Container Toolkit，讓容器能存取 GPU；映像檔中的 CUDA 版本要與驅動相容。",
      "最佳實務包含固定相依版本、使用多階段建置縮小映像、不把機密寫入映像，並以標籤或雜湊管理版本。",
    ],
    diagram: [
      "Dockerfile",
      "建置映像檔",
      "推送到映像檔倉庫",
      "在目標環境拉取",
      "啟動容器（可掛載 GPU）",
    ],
    code: `# Dockerfile 示意
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "serve.py"]`,
    caseTitle: "Docker 與 NVIDIA Container Toolkit",
    case: "Docker 官方文件說明了容器與映像檔的基礎，NVIDIA Container Toolkit 文件則說明容器如何使用 GPU。具體版本相容性請查閱各自文件。",
    related: ["kubernetes", "model-serving", "cuda", "workflow-orchestration"],
    sources: [
      ["Docker get started", "https://docs.docker.com/get-started/"],
      ["NVIDIA Container Toolkit", "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html"],
    ],
  },
  {
    id: "kubernetes",
    title: "Kubernetes",
    zh: "Kubernetes 容器編排",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "管理容器化應用的開源平台，提供排程、擴縮容、服務發現與自我修復，常用於部署模型服務與訓練工作。",
    intuition:
      "像大型貨櫃碼頭的調度中心：決定每個貨櫃（容器）放到哪個碼頭（節點）、壞了自動換一個、貨量大時加開更多位置。",
    mechanism: [
      "叢集由控制平面與多個工作節點組成；Pod 是最小部署單位，包含一或多個容器。",
      "以 Deployment 宣告期望的副本數與映像檔，控制器持續比對實際與期望狀態，並自動重啟或補足 Pod。",
      "Service 提供穩定的網路入口與負載平衡；HPA（Horizontal Pod Autoscaler）可依 CPU 或自訂指標自動調整副本數。",
      "使用 GPU 時需要安裝裝置外掛（device plugin），並在 Pod 中以資源請求指定 GPU 數量；GPU 的共享、拓樸與排程是進階議題。",
    ],
    diagram: [
      "提交 Deployment 宣告",
      "控制平面排程 Pod 到節點",
      "容器啟動（可請求 GPU）",
      "Service 負載平衡",
      "HPA 依指標擴縮",
    ],
    code: `# Pod 請求 GPU 的 YAML 片段（需已安裝 GPU device plugin）
resources:
  limits:
    nvidia.com/gpu: 1`,
    caseTitle: "Kubernetes 的 GPU 排程",
    case: "Kubernetes 官方文件說明了核心概念與以裝置外掛排程 GPU 的方式。在 Kubernetes 上跑機器學習常搭配 KServe、Kubeflow 等專案，但維運複雜度需要納入評估。",
    related: ["containerization", "model-serving", "workflow-orchestration", "ml-observability"],
    sources: [
      ["Kubernetes overview", "https://kubernetes.io/docs/concepts/overview/"],
      ["Kubernetes: scheduling GPUs", "https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/"],
    ],
  },
  {
    id: "triton-inference-server",
    title: "Triton Inference Server",
    zh: "Triton 推論伺服器",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "NVIDIA 的開源推論服務軟體，支援多種模型格式與後端，提供動態批次、並行執行與模型管理。",
    intuition:
      "像一個能同時服務多種模型的收發中心：不管模型是用哪個框架做的，都放進統一的櫃台，由它負責排隊、合併請求與分配 GPU。",
    mechanism: [
      "模型放在模型倉庫（model repository），每個模型有設定檔（config.pbtxt）說明輸入輸出、後端與批次策略。",
      "支援多種後端，例如 TensorRT、ONNX Runtime、PyTorch、Python 後端等，並可同時載入多個模型與多個版本。",
      "動態批次器（dynamic batcher）把短時間內到達的請求合併成較大的批次，以提升 GPU 使用率，並可設定最大等待時間來控制延遲。",
      "對外提供 HTTP 與 gRPC 端點與指標（可由 Prometheus 蒐集）；與 TensorRT-LLM 等後端搭配可服務大型語言模型。",
    ],
    diagram: [
      "客戶端（HTTP / gRPC）",
      "請求排程與動態批次",
      "後端（TensorRT / ONNX / PyTorch…）",
      "GPU / CPU 執行",
      "回傳結果與指標",
    ],
    code: `# config.pbtxt 片段（示意；欄位以官方文件為準）
# name: "my_model"
# backend: "onnxruntime"
# max_batch_size: 16
# dynamic_batching { max_queue_delay_microseconds: 1000 }`,
    caseTitle: "官方文件與開源專案",
    case: "NVIDIA 公開了 Triton 的使用者指南與 GitHub 專案，說明模型倉庫與動態批次等功能。是否適用取決於模型類型與基礎設施，亦可與其他服務框架比較。",
    related: ["model-serving", "tensorrt", "onnx-runtime", "kubernetes"],
    sources: [
      ["Triton Inference Server user guide", "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/"],
      ["Triton Inference Server repository", "https://github.com/triton-inference-server/server"],
    ],
  },
  {
    id: "ml-observability",
    title: "ML Observability",
    zh: "機器學習系統可觀測性",
    category: "Data & MLOps",
    level: "Core focus",
    status: "published",
    summary:
      "透過指標、日誌與追蹤（traces），再加上資料與模型品質的監控，掌握線上機器學習系統是否健康、為何變差。",
    intuition:
      "像醫院的監控儀表板：不只看機器有沒有開（系統是否運作），也要看病人的指標有沒有變差（模型預測是否仍然可靠）。",
    mechanism: [
      "系統層：請求量、延遲分位數、錯誤率、GPU 使用率與記憶體，常以 Prometheus 與 Grafana 等蒐集與視覺化。",
      "追蹤層：以分散式追蹤（如 OpenTelemetry）串起一次請求經過的各個元件，找出延遲與錯誤的來源；對 LLM 應用可記錄提示、工具呼叫與每個步驟的耗時與 token 用量。",
      "資料與模型層：輸入分佈、預測分佈、資料漂移與（標籤延遲到達後的）實際準確度與業務指標。",
      "警報要設計成可行動：過多的假警報會被忽略；日誌與追蹤也要注意隱私，避免把個資或機密寫入。",
    ],
    diagram: [
      "服務輸出指標、日誌、追蹤",
      "蒐集並儲存",
      "儀表板與查詢",
      "警報規則",
      "調查並修復",
    ],
    code: `from prometheus_client import Histogram

LATENCY = Histogram("inference_latency_seconds", "Inference latency")

@LATENCY.time()
def handle(request):
    return model(request)`,
    caseTitle: "OpenTelemetry 與 Prometheus",
    case: "OpenTelemetry 與 Prometheus 是常見的開源可觀測性工具，官方文件說明了追蹤、指標的蒐集方式。機器學習特有的監控（如漂移）常需要額外的工具或自訂指標。",
    related: ["data-drift", "model-serving", "latency-throughput", "kubernetes"],
    sources: [
      ["OpenTelemetry documentation", "https://opentelemetry.io/docs/"],
      ["Prometheus overview", "https://prometheus.io/docs/introduction/overview/"],
    ],
  },
];
