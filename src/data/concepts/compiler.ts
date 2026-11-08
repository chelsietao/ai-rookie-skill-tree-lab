import type { Concept } from "@/types";

export const compilerConcepts: Concept[] = [
  {
    id: "xla",
    title: "XLA",
    zh: "加速線性代數編譯器",
    category: "Compiler Engineering",
    level: "Core focus",
    status: "published",
    summary:
      "OpenXLA 專案中的機器學習編譯器，把計算圖（HLO）最佳化並編譯成 CPU、GPU 或 TPU 的可執行程式。",
    intuition:
      "像把一份逐步指示的食譜重新整理成流水線：先看完整份食譜，把能合併的步驟合併、把多餘的備料省掉，再交給不同廚房（硬體）照做。",
    mechanism: [
      "前端框架（例如 JAX、TensorFlow，或 PyTorch/XLA）把計算表示成 HLO（High Level Operations）圖。",
      "XLA 在 HLO 上做與硬體無關的最佳化，例如代數化簡、常數折疊、operator fusion 與記憶體配置規劃。",
      "接著由各後端（CPU、GPU、TPU 等）做硬體相關的最佳化並產生程式碼。",
      "編譯結果會針對特定形狀與型態特化，形狀變化大的工作負載需要注意重新編譯的成本。",
    ],
    diagram: [
      "框架程式（JAX / TF / PyTorch-XLA）",
      "HLO 計算圖",
      "硬體無關最佳化（fusion 等）",
      "後端程式碼生成",
      "可執行檔（CPU / GPU / TPU）",
    ],
    code: `# JAX 以 jit 觸發 XLA 編譯
import jax, jax.numpy as jnp

@jax.jit
def f(x, w):
    return jnp.tanh(x @ w).sum()

x = jnp.ones((128, 256)); w = jnp.ones((256, 64))
print(f(x, w))   # 首次呼叫會編譯，之後重用`,
    caseTitle: "JAX 與 OpenXLA",
    case: "JAX 文件說明 jax.jit 會透過 XLA 把函式編譯為特定硬體的程式；OpenXLA 專案則公開了 XLA 的架構與文件。具體的加速幅度取決於模型與硬體，需要自行量測。",
    related: ["operator-fusion", "computation-graph", "mlir", "torch-compile"],
    sources: [
      ["OpenXLA: XLA", "https://openxla.org/xla"],
      ["JAX: Just-in-time compilation", "https://docs.jax.dev/en/latest/jit-compilation.html"],
    ],
  },
  {
    id: "mlir",
    title: "MLIR",
    zh: "多層次中間表示",
    category: "Compiler Engineering",
    level: "Reference",
    status: "published",
    summary:
      "LLVM 生態中的可擴充編譯器基礎設施，以 dialect 與多層次 IR 讓不同抽象層級的最佳化能在同一框架下進行。",
    intuition:
      "像一套可以自訂語言層級的翻譯流程：從「張量運算」逐層降到「迴圈」再到「機器指令」，每一層都有自己熟悉的最佳化手法。",
    mechanism: [
      "MLIR 以 operation、region、type 與 attribute 為基本結構，使用者可自訂 dialect 來表示特定領域的運算。",
      "不同 dialect 可以共存於同一份 IR，編譯器透過 lowering（降階）與 pattern rewrite 逐步把高階運算轉成低階表示。",
      "這種設計讓高階的圖最佳化（例如 tiling、fusion）與低階的程式碼生成可以共用基礎設施，而不必各自重建整套編譯器。",
      "許多機器學習編譯器與加速器工具鏈（包含 OpenXLA 相關工作）採用 MLIR，但各專案選用的 dialect 與流程不同。",
    ],
    diagram: [
      "高階 dialect（張量運算）",
      "圖層最佳化 / fusion",
      "Lowering 到迴圈與記憶體 dialect",
      "Lowering 到 LLVM / 硬體 dialect",
      "機器碼或加速器程式",
    ],
    code: `// MLIR 文字格式示意（簡化）
func.func @add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}`,
    caseTitle: "MLIR 論文",
    case: "MLIR 論文闡述了以可擴充 dialect 建立統一編譯器基礎設施的動機與設計；LLVM 官方網站提供其文件與教學。各專案如何使用 MLIR 以其自身文件為準。",
    related: ["xla", "computation-graph", "operator-fusion", "triton"],
    sources: [
      ["MLIR paper", "https://arxiv.org/abs/2002.11054"],
      ["MLIR documentation", "https://mlir.llvm.org/"],
    ],
  },
  {
    id: "onnx",
    title: "ONNX",
    zh: "開放神經網路交換格式",
    category: "Compiler Engineering",
    level: "Reference",
    status: "published",
    summary:
      "用於表示機器學習模型的開放格式，定義計算圖、運算子（operators）與資料型態，方便跨框架與跨執行環境交換模型。",
    intuition:
      "像模型界的通用交換檔案：在 PyTorch 訓練好之後，可以輸出成標準格式，交給不同的執行引擎使用。",
    mechanism: [
      "ONNX 模型是一張有向計算圖，節點是運算子，邊是張量，並記錄輸入輸出的型態與（可選的）形狀資訊。",
      "運算子集合以版本化的 opset 管理；匯出時需選擇相容的 opset，並確認目標執行環境支援這些運算子。",
      "PyTorch 等框架提供匯出介面，把模型追蹤或轉換成 ONNX；動態控制流程與自訂運算子可能需要額外處理。",
      "匯出後常搭配 ONNX Runtime、TensorRT 等引擎做最佳化與部署，並以數值比對確認匯出前後的輸出一致。",
    ],
    diagram: [
      "框架模型（PyTorch 等）",
      "匯出成 ONNX 圖（指定 opset）",
      "圖驗證與最佳化",
      "交給執行引擎",
      "部署與數值比對",
    ],
    code: `import torch

model = torch.nn.Linear(16, 4).eval()
dummy = torch.randn(1, 16)
torch.onnx.export(
    model, (dummy,), "linear.onnx",
    input_names=["x"], output_names=["y"],
    dynamic_axes={"x": {0: "batch"}},
)`,
    caseTitle: "跨框架模型交換",
    case: "ONNX 官方文件描述了其作為跨框架互通格式的設計與運算子規格；PyTorch 文件也提供匯出到 ONNX 的流程。匯出是否成功、效能是否提升，取決於模型結構與目標執行環境。",
    related: ["onnx-runtime", "tensorrt", "computation-graph", "model-serving"],
    sources: [
      ["ONNX documentation", "https://onnx.ai/onnx/"],
      ["PyTorch: torch.onnx", "https://docs.pytorch.org/docs/stable/onnx.html"],
    ],
  },
  {
    id: "onnx-runtime",
    title: "ONNX Runtime",
    zh: "ONNX 執行引擎",
    category: "Compiler Engineering",
    level: "Reference",
    status: "published",
    summary:
      "執行 ONNX 模型的跨平台推論引擎，透過 execution provider 對接 CPU、CUDA、TensorRT 等不同硬體後端。",
    intuition:
      "像一個通用的播放器：同一份 ONNX 檔案放進去，可以選擇用不同的「解碼硬體」播放，播放器負責做圖層最佳化與分派。",
    mechanism: [
      "載入 ONNX 圖後，ONNX Runtime 會做圖最佳化，例如常數折疊與節點融合，不同等級可以由設定調整。",
      "Execution provider（EP）宣告自己能處理圖中的哪些子圖，引擎把子圖分配給最合適的 EP，其餘回退到 CPU。",
      "同一模型可在不同 EP 之間切換（例如 CPU、CUDA、TensorRT），便於跨環境部署。",
      "實際效能取決於模型運算子是否被 EP 支援、是否發生子圖切分與資料搬移，需要以 profiling 驗證。",
    ],
    diagram: [
      "載入 ONNX 模型",
      "圖最佳化",
      "依 execution provider 切分子圖",
      "各 EP 執行子圖",
      "輸出結果",
    ],
    code: `import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "linear.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
y = sess.run(None, {"x": np.random.randn(1, 16).astype("float32")})`,
    caseTitle: "Execution provider 機制",
    case: "ONNX Runtime 文件說明其以 execution provider 擴充硬體支援，並列出可用的 EP。是否能獲得加速需看模型與 EP 的支援程度。",
    related: ["onnx", "tensorrt", "model-serving", "latency-throughput"],
    sources: [
      ["ONNX Runtime documentation", "https://onnxruntime.ai/docs/"],
      ["ONNX Runtime execution providers", "https://onnxruntime.ai/docs/execution-providers/"],
    ],
  },
  {
    id: "tensorrt",
    title: "TensorRT",
    zh: "NVIDIA 推論最佳化器",
    category: "Compiler Engineering",
    level: "Core focus",
    status: "published",
    summary:
      "NVIDIA 的深度學習推論最佳化與執行函式庫，針對 NVIDIA GPU 做圖最佳化、kernel 選擇與低精度推論，產出可序列化的 engine。",
    intuition:
      "像針對特定賽道（某一代 GPU）預先調校的賽車：先花時間試遍各種設定，把最快的組合固定下來，之後上路就不用再摸索。",
    mechanism: [
      "輸入常透過 ONNX 等格式匯入網路定義，TensorRT 依此建立內部的網路表示。",
      "Builder 階段會做層融合、常數折疊，並針對目標 GPU 嘗試多種 kernel 實作（tactic），選出較快者。",
      "可啟用 FP16、INT8 或 FP8 等低精度模式以提高吞吐，部分模式需要校準資料並評估精度影響。",
      "輸出的 engine 與 GPU 型號及 TensorRT 版本相關，通常需要在目標環境上建置，並要處理動態形狀的 optimization profile。",
    ],
    diagram: [
      "ONNX / 網路定義",
      "Builder：融合與精度選擇",
      "Tactic 試驗與 kernel 選擇",
      "序列化 engine",
      "Runtime 載入並執行",
    ],
    code: `# 以官方 trtexec 工具從 ONNX 建立 engine（參數以官方文件為準）
# trtexec --onnx=model.onnx --saveEngine=model.plan --fp16`,
    caseTitle: "TensorRT-LLM 與 Triton Inference Server",
    case: "NVIDIA 公開了 TensorRT 的文件，並另有 TensorRT-LLM 專案針對大型語言模型推論；engine 常搭配 Triton Inference Server 部署。效能數字隨 GPU 與版本不同，請參考官方基準與自行量測。",
    related: ["onnx", "onnx-runtime", "quantization", "cuda"],
    sources: [
      ["TensorRT documentation", "https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html"],
      ["TensorRT-LLM repository", "https://github.com/NVIDIA/TensorRT-LLM"],
    ],
  },
  {
    id: "torch-inductor",
    title: "TorchInductor",
    zh: "PyTorch 預設編譯後端",
    category: "Compiler Engineering",
    level: "Core focus",
    status: "published",
    summary:
      "torch.compile 預設使用的編譯後端，把捕捉到的 FX 圖降階並生成 GPU 的 Triton kernel 或 CPU 的 C++/OpenMP 程式碼。",
    intuition:
      "像 torch.compile 背後的廚房：前端（Dynamo）負責把食譜抄成清單，Inductor 則決定哪些步驟合併、並直接寫出能在顯卡上跑的程式。",
    mechanism: [
      "TorchDynamo 攔截 Python 位元碼並捕捉出 FX 圖，遇到無法捕捉的部分會產生 graph break。",
      "AOTAutograd 追蹤前向與反向圖，讓兩者都能被編譯。",
      "Inductor 把圖降階為自己的 IR，做 operator fusion、排程與記憶體規劃。",
      "在 NVIDIA GPU 上主要生成 Triton kernel，在 CPU 上生成 C++ 程式碼；編譯結果會被快取，不同形狀可能觸發重新編譯。",
    ],
    diagram: [
      "Python 模型程式",
      "TorchDynamo 捕捉 FX 圖",
      "AOTAutograd 前向／反向圖",
      "Inductor 降階與融合",
      "Triton / C++ kernel",
    ],
    code: `import torch

model = torch.nn.Sequential(
    torch.nn.Linear(256, 256), torch.nn.GELU()
).cuda()
compiled = torch.compile(model)   # 預設 backend="inductor"
y = compiled(torch.randn(32, 256, device="cuda"))`,
    caseTitle: "PyTorch 2 編譯堆疊",
    case: "PyTorch 官方文件說明 torch.compile 的元件與預設後端，PyTorch 開發者論壇也有 TorchInductor 的設計文章。效益會因模型與 graph break 數量而異，需自行以基準測試確認。",
    related: ["torch-compile", "operator-fusion", "triton", "computation-graph"],
    sources: [
      ["PyTorch torch.compile overview", "https://docs.pytorch.org/docs/stable/torch.compiler.html"],
      ["TorchInductor design post", "https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747"],
    ],
  },
];
