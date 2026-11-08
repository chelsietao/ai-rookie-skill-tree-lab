import type { Concept, ConceptPaper } from "@/types";

type Draft = Omit<Concept, "category" | "status" | "level"> & {
  level?: Concept["level"];
};

const c = (draft: Draft): Concept => ({
  level: "Reference",
  ...draft,
  category: "Compiler Engineering",
  status: "published",
});

const p = (
  title: string,
  venue: string,
  year: number,
  url: string,
  note: string
): ConceptPaper => ({ title, venue, year, url, note });

const MLIR_PAPER = (note: string) =>
  p(
    "MLIR: A Compiler Infrastructure for the End of Moore's Law",
    "arXiv",
    2020,
    "https://arxiv.org/abs/2002.11054",
    note
  );
const PT2_PAPER = (note: string) =>
  p(
    "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation",
    "ASPLOS",
    2024,
    "https://doi.org/10.1145/3620665.3640366",
    note
  );
const LLVM_PAPER = (note: string) =>
  p(
    "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation",
    "CGO",
    2004,
    "https://doi.org/10.1109/CGO.2004.1281665",
    note
  );
const TVM_PAPER = (note: string) =>
  p(
    "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning",
    "OSDI",
    2018,
    "https://arxiv.org/abs/1802.04799",
    note
  );

const LLVM_PASSES: [string, string] = [
  "LLVM: Passes reference",
  "https://llvm.org/docs/Passes.html",
];

export const compilerExtendedConcepts: Concept[] = [
  c({
    id: "llvm",
    title: "LLVM",
    zh: "LLVM 編譯器基礎設施",
    level: "Core focus",
    summary: "模組化的編譯器基礎設施，以共通的中間表示連接多種語言前端與硬體後端。",
    intuition: "像國際機場的轉運中心。不同語言先轉成同一種中介格式，再轉機飛往不同的硬體。",
    mechanism: [
      "前端（例如 Clang）把原始碼轉成 LLVM IR。",
      "中端以一系列 pass 對 IR 進行與硬體無關的最佳化。",
      "後端完成指令選擇、暫存器配置與排程，並產生機器碼。",
    ],
    diagram: ["原始碼", "前端產生 LLVM IR", "最佳化 passes", "後端", "機器碼"],
    code: `clang -O2 -S -emit-llvm add.c -o add.ll  # 產生 LLVM IR
opt -passes=mem2reg add.ll -S -o out.ll   # 執行最佳化 pass
llc out.ll -o add.s                        # 產生組合語言`,
    caseTitle: "機器學習工具鏈的底層",
    case: "XLA 的 CPU 與 GPU 後端、Triton 和 MLIR 的底層程式碼生成，都會使用 LLVM。",
    related: ["ssa-form", "mlir", "xla", "llvm-pass-manager"],
    sources: [["LLVM documentation", "https://llvm.org/docs/"]],
    papers: [LLVM_PAPER("LLVM 的原始論文。")],
  }),
  c({
    id: "ssa-form",
    title: "SSA Form",
    zh: "靜態單賦值形式",
    summary: "每個變數只被賦值一次的中間表示形式，讓資料流分析與最佳化變得簡單。",
    intuition: "每次更新數值都換一張新標籤，看到標籤就知道它來自哪裡，不必追溯整段程式。",
    mechanism: [
      "每次賦值都會產生新版本的變數，例如 x1、x2。",
      "控制流匯合處以 phi 節點選擇來自不同路徑的值。",
      "使用與定義的關係清楚，常數傳播與死碼消除等分析都因此變得容易。",
    ],
    diagram: ["一般程式碼", "變數重新命名", "插入 phi 節點", "SSA IR"],
    code: `; LLVM IR 的 phi 節點
entry:
  br i1 %cond, label %a, label %b
a:
  %x1 = add i32 1, 2
  br label %merge
b:
  %x2 = add i32 3, 4
  br label %merge
merge:
  %x3 = phi i32 [ %x1, %a ], [ %x2, %b ]`,
    caseTitle: "編譯器的共通做法",
    case: "LLVM IR 與 MLIR 都以 SSA 形式表示值，而 GCC 的中端 GIMPLE 也有 SSA 形式。",
    related: ["llvm", "mlir", "constant-folding", "dead-code-elimination"],
    sources: [["LLVM Language Reference", "https://llvm.org/docs/LangRef.html"]],
    papers: [p("Efficiently Computing Static Single Assignment Form and the Control Dependence Graph", "TOPLAS", 1991, "https://doi.org/10.1145/115372.115320", "提出有效建構 SSA 的方法。")],
  }),
  c({
    id: "apache-tvm",
    title: "Apache TVM",
    zh: "Apache TVM 編譯器",
    summary: "把多種框架的模型編譯成可部署到 CPU、GPU 與加速器的最佳化程式，並以機器學習輔助調校。",
    intuition: "像智慧型裁縫，量好不同硬體的身形，自動試版，做出最合身的版本。",
    mechanism: [
      "匯入模型後轉成高階圖 IR，並進行融合等最佳化。",
      "以排程原語描述迴圈的切分與平行方式。",
      "AutoTVM 或 Ansor 會搜尋排程，並以成本模型挑選較好的實作。",
    ],
    diagram: ["ONNX / PyTorch 模型", "圖層 IR", "張量層 IR 與排程", "程式碼生成", "部署"],
    code: `import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# 實際介面請以所安裝版本的文件為準
mod = from_onnx(onnx_model)
print(mod)`,
    caseTitle: "硬體與軟體共同研究",
    case: "TVM 起源於華盛頓大學的研究，之後成為 Apache 專案，並用於多種硬體的模型編譯研究與部署。",
    related: ["ansor", "learned-cost-model", "operator-fusion", "halide"],
    sources: [["Apache TVM documentation", "https://tvm.apache.org/docs/"]],
    papers: [TVM_PAPER("TVM 的原始論文。")],
  }),
  c({
    id: "glow",
    title: "Glow",
    zh: "Glow 機器學習編譯器",
    summary: "Meta 開發的機器學習編譯器，以兩層 IR 進行最佳化，並支援異質硬體。",
    intuition: "先從流程圖看哪些工序可以合併，再細到每一步用哪個容器，避免浪費。",
    mechanism: [
      "高階 IR 保留運算子圖，便於做圖最佳化。",
      "低階 IR 以記憶體位址為主，便於規劃記憶體與移除複製。",
      "後端再針對各硬體生成程式。",
    ],
    diagram: ["模型", "高階圖 IR", "低階指令 IR", "硬體後端"],
    code: `# 使用 Glow 的 model-compiler 工具（參數請見官方文件）
model-compiler -backend=CPU -model=model.onnx -emit-bundle=build/`,
    caseTitle: "Glow 的設計",
    case: "Glow 論文介紹以兩階段 IR 把模型降階，並為硬體廠商提供擴充的介面。",
    related: ["apache-tvm", "static-memory-planning", "onnx", "operator-fusion"],
    sources: [["Glow (GitHub)", "https://github.com/pytorch/glow"]],
    papers: [p("Glow: Graph Lowering Compiler Techniques for Neural Networks", "arXiv", 2018, "https://arxiv.org/abs/1805.00907", "說明兩層 IR 與降階流程。")],
  }),
  c({
    id: "iree",
    title: "IREE",
    zh: "IREE 編譯器與執行環境",
    summary: "以 MLIR 為基礎的端到端編譯器與輕量執行環境，可部署到行動裝置、嵌入式與資料中心。",
    intuition: "把大型廚房的流程精簡成行動餐車，到哪都能照同一份流程出餐。",
    mechanism: [
      "匯入 StableHLO、Torch 等輸入，再透過 MLIR 逐層降階。",
      "把計算切分成可排程的工作單元，產生 GPU、CPU 等後端程式。",
      "輕量的 HAL runtime 負責裝置抽象與執行。",
    ],
    diagram: ["輸入模型", "MLIR 降階", "Flow / Stream 排程", "HAL 後端", "輕量 runtime"],
    code: `# 把 MLIR 編譯成可在 CPU 上執行的模組
iree-compile model.mlir \\
  --iree-hal-target-backends=llvm-cpu \\
  -o model.vmfb
iree-run-module --module=model.vmfb --function=main`,
    caseTitle: "OpenXLA 生態",
    case: "IREE 是開源專案，文件中說明了從多種機器學習框架到不同硬體的編譯路徑。",
    related: ["mlir", "stablehlo", "linalg-dialect", "executorch"],
    sources: [["IREE documentation", "https://iree.dev/"]],
    papers: [MLIR_PAPER("IREE 建構於 MLIR 之上。")],
  }),
  c({
    id: "linalg-dialect",
    title: "Linalg Dialect",
    zh: "MLIR Linalg 方言",
    summary: "MLIR 中描述結構化線性代數運算的方言，便於進行 tiling、fusion 與向量化。",
    intuition: "不直接寫三層迴圈，而是聲明「這是矩陣相乘」。編譯器知道意圖後，才容易切塊與重排。",
    mechanism: [
      "linalg.generic 以 indexing maps 與 iterator types 描述運算的結構。",
      "這種結構讓 tiling、fusion 和 vectorization 成為通用的轉換。",
      "之後可降階到迴圈或向量方言。",
    ],
    diagram: ["Tensor 方言", "Linalg 結構化運算", "Tiling / Fusion", "向量化或迴圈", "LLVM"],
    code: `// 矩陣相乘
%result = linalg.matmul
  ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
  outs(%C : tensor<4x16xf32>) -> tensor<4x16xf32>`,
    caseTitle: "MLIR 編譯器的中介層",
    case: "IREE 等 MLIR 工具鏈常以 Linalg 作為程式碼生成的關鍵中介層。",
    related: ["mlir", "loop-tiling", "loop-fusion", "mlir-affine"],
    sources: [["MLIR: Linalg Dialect", "https://mlir.llvm.org/docs/Dialects/Linalg/"]],
    papers: [p("Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction", "arXiv", 2022, "https://arxiv.org/abs/2202.03293", "說明以結構化運算建構張量編譯器的方法。")],
  }),
  c({
    id: "stablehlo",
    title: "StableHLO",
    zh: "StableHLO 運算集合",
    summary: "OpenXLA 維護的可攜運算集合，作為機器學習框架與編譯器之間穩定的交換格式。",
    intuition: "像貨櫃的統一規格。不管是哪家公司出貨、哪個港口收貨，都能順利對接。",
    mechanism: [
      "以 MLIR 方言定義，並有明確的規格與相容性保證。",
      "JAX 等框架可以輸出 StableHLO，再交給編譯器處理。",
      "版本機制讓不同時間產生的程式仍能被處理。",
    ],
    diagram: ["JAX / PyTorch / TF", "StableHLO", "XLA / IREE 等編譯器", "硬體"],
    code: `import jax, jax.numpy as jnp

def f(x):
    return jnp.sin(x) * 2

print(jax.jit(f).lower(jnp.ones(3)).as_text())  # 印出 StableHLO`,
    caseTitle: "框架與編譯器之間",
    case: "OpenXLA 以 StableHLO 作為框架與編譯器間的介面，JAX 的 lower 介面也能直接印出它的內容。",
    related: ["xla", "mlir", "iree", "torch-mlir"],
    sources: [["OpenXLA: StableHLO", "https://openxla.org/stablehlo"]],
    papers: [MLIR_PAPER("StableHLO 以 MLIR 方言定義，這篇說明其底層架構。")],
  }),
  c({
    id: "torch-mlir",
    title: "Torch-MLIR",
    zh: "Torch-MLIR 連接層",
    summary: "把 PyTorch 模型轉成 MLIR 的開源專案，並能進一步降階到 Linalg、TOSA 或 StableHLO。",
    intuition: "在 PyTorch 與 MLIR 兩個世界之間的翻譯官。",
    mechanism: [
      "以 torch 方言對應 PyTorch 的運算子。",
      "經過形狀與型別推斷等處理，把程式轉成較好處理的形式。",
      "再降階到下游編譯器可接受的方言。",
    ],
    diagram: ["PyTorch 程式", "torch 方言", "Linalg / TOSA / StableHLO", "下游編譯器"],
    code: `# 使用方式依版本不同，建議參考官方範例
# 主要流程：torch.export -> torch 方言 -> linalg-on-tensors
python -m torch_mlir.tools.import_onnx model.onnx -o model.mlir`,
    caseTitle: "連接 PyTorch 與硬體工具鏈",
    case: "許多以 MLIR 為基礎的工具鏈透過 Torch-MLIR 來接收 PyTorch 模型。",
    related: ["mlir", "stablehlo", "linalg-dialect", "iree"],
    sources: [["Torch-MLIR (GitHub)", "https://github.com/llvm/torch-mlir"]],
    papers: [MLIR_PAPER("Torch-MLIR 以 MLIR 為目標表示。")],
  }),
  c({
    id: "torchscript",
    title: "TorchScript",
    zh: "TorchScript 中間表示",
    summary: "PyTorch 早期的模型序列化與 IR，可以透過追蹤或腳本化，把模型移到 Python 之外執行。",
    intuition: "把即興演出整理成劇本，之後即使沒有導演，也能照劇本演出。",
    mechanism: [
      "torch.jit.trace 以範例輸入記錄實際執行的運算，動態控制流可能被固定下來。",
      "torch.jit.script 解析 Python 子集，因此能保留控制流。",
      "輸出的模型可由 C++ 的 LibTorch 載入。",
    ],
    diagram: ["PyTorch 模型", "trace 或 script", "TorchScript IR", "LibTorch 執行"],
    code: `import torch

scripted = torch.jit.script(model)
scripted.save("model.pt")
loaded = torch.jit.load("model.pt")`,
    caseTitle: "逐漸由新工具接手",
    case: "PyTorch 2 之後，官方的編譯與匯出重心逐漸轉向 torch.compile 與 torch.export，但既有部署仍可能使用 TorchScript。",
    related: ["torch-compile", "torch-fx", "tracing-jit", "torch-dynamo"],
    sources: [["PyTorch: TorchScript", "https://pytorch.org/docs/stable/jit.html"]],
    papers: [p("PyTorch: An Imperative Style, High-Performance Deep Learning Library", "NeurIPS", 2019, "https://arxiv.org/abs/1912.01703", "說明 PyTorch 的設計，以及動態圖帶來的取捨。")],
  }),
  c({
    id: "torch-fx",
    title: "torch.fx",
    zh: "torch.fx 圖轉換工具",
    summary: "以 Python 撰寫 PyTorch 模型圖轉換的工具，可把模型符號追蹤成可修改的圖。",
    intuition: "把機器的接線圖攤開，用 Python 改接線，再把它重新組回機器。",
    mechanism: [
      "symbolic tracer 以 proxy 物件記錄運算，形成 Graph。",
      "Graph 由節點構成，使用者可以新增、替換或刪除節點。",
      "修改後重新產生 Python 程式，並組成 GraphModule。",
    ],
    diagram: ["nn.Module", "Symbolic Trace", "FX Graph", "圖轉換", "GraphModule"],
    code: `import torch.fx as fx

gm = fx.symbolic_trace(model)
for node in gm.graph.nodes:
    print(node.op, node.target)
gm.recompile()`,
    caseTitle: "圖模式量化的基礎",
    case: "torch.fx 是 PyTorch 圖模式工具的基礎之一，TorchDynamo 擷取的圖也以 FX Graph 表示。",
    related: ["torch-dynamo", "aot-autograd", "torch-inductor", "computation-graph"],
    sources: [["PyTorch: torch.fx", "https://pytorch.org/docs/stable/fx.html"]],
    papers: [p("torch.fx: Practical Program Capture and Transformation for Deep Learning in Python", "MLSys", 2022, "https://arxiv.org/abs/2112.08429", "torch.fx 的設計與取捨。")],
  }),
  c({
    id: "torch-dynamo",
    title: "TorchDynamo",
    zh: "TorchDynamo 圖擷取",
    level: "Core focus",
    summary: "torch.compile 的前端，在 Python 位元組碼層級擷取 FX 圖，無法處理的部分則退回一般執行。",
    intuition: "同聲傳譯員聽演講時，遇到能翻譯的段落就交給機器，遇到不懂的就讓講者自己講。",
    mechanism: [
      "透過 PEP 523 的 frame evaluation API 攔截 Python 函式執行。",
      "分析位元組碼，把 PyTorch 運算擷取成 FX 圖。",
      "遇到無法追蹤的情況會產生 graph break，同時以 guard 檢查假設是否仍成立。",
    ],
    diagram: ["Python 位元組碼", "Frame 攔截", "FX 圖與 guard", "編譯子圖", "Graph break 退回 Python"],
    code: `import torch

def fn(x):
    return torch.relu(x).sum()

print(torch._dynamo.explain(fn)(torch.ones(3)))  # 檢查 graph break`,
    caseTitle: "PyTorch 2 的前端",
    case: "PyTorch 2 論文說明 TorchDynamo 如何讓既有的動態程式在幾乎不改動的情況下使用編譯。",
    related: ["torch-compile", "torch-fx", "aot-autograd", "torch-inductor"],
    sources: [["PyTorch: torch.compiler", "https://pytorch.org/docs/stable/torch.compiler.html"]],
    papers: [PT2_PAPER("說明 TorchDynamo 與 Python 位元組碼轉換。")],
  }),
  c({
    id: "aot-autograd",
    title: "AOTAutograd",
    zh: "AOTAutograd 預先反向圖",
    summary: "在編譯時預先產生反向傳播的圖，讓前向與反向可以一起最佳化。",
    intuition: "開工前就把來回路線都規劃好，而不是到了目的地才想怎麼回來。",
    mechanism: [
      "把 FX 圖追蹤到 PyTorch 的 dispatcher，並產生前向與反向圖。",
      "把運算分解成較小的運算子集合，便於後端處理。",
      "以 min-cut 等方法決定前向與反向之間要保存哪些值。",
    ],
    diagram: ["Dynamo 的 FX 圖", "追蹤 autograd", "前向與反向圖", "分解成小運算", "交給 Inductor"],
    code: `# 使用者通常不會直接呼叫，torch.compile 會自動使用
import torch

compiled = torch.compile(model)
loss = compiled(x).sum()
loss.backward()`,
    caseTitle: "torch.compile 的中段",
    case: "在 torch.compile 的流程中，AOTAutograd 位於 TorchDynamo 與 TorchInductor 之間。",
    related: ["torch-dynamo", "torch-inductor", "autograd", "enzyme-ad"],
    sources: [["PyTorch: torch.compiler", "https://pytorch.org/docs/stable/torch.compiler.html"]],
    papers: [PT2_PAPER("說明 AOTAutograd 在編譯流程中的角色。")],
  }),
  c({
    id: "executorch",
    title: "ExecuTorch",
    zh: "ExecuTorch 端側執行環境",
    summary: "PyTorch 面向行動與邊緣裝置的執行環境，透過 torch.export 轉換模型，並交由硬體後端執行。",
    intuition: "把大型卡車上的貨物，重新打包成機車也能送的小包裹。",
    mechanism: [
      "以 torch.export 把模型匯出成圖。",
      "把可支援的子圖委派給 XNNPACK、Core ML 或 QNN 等後端。",
      "輸出 .pte 檔，由輕量 C++ runtime 執行。",
    ],
    diagram: ["PyTorch 模型", "torch.export", "Edge 方言", "後端委派", ".pte 與 runtime"],
    code: `import torch
from executorch.exir import to_edge

ep = torch.export.export(model, (example,))
program = to_edge(ep).to_executorch()
with open("model.pte", "wb") as f:
    f.write(program.buffer)`,
    caseTitle: "PyTorch 的端側部署",
    case: "ExecuTorch 是 PyTorch 官方的端側方案，文件列出多種硬體後端。",
    related: ["xnnpack", "torch-compile", "quantization", "iree"],
    sources: [["ExecuTorch documentation", "https://docs.pytorch.org/executorch/stable/index.html"]],
  }),
  c({
    id: "polyhedral-model",
    title: "Polyhedral Model",
    zh: "多面體模型",
    summary: "把巢狀迴圈表示成數學上的多面體，再以線性代數尋找合法的重排方式。",
    intuition: "把迴圈的每次執行都當成座標上的點，整段程式就是一個形狀，重排等於換一個角度切這個形狀。",
    mechanism: [
      "以不等式描述迴圈的迭代空間。",
      "分析相依關係，確保重排不改變程式結果。",
      "求解排程以改善平行度與快取區域性，但可能花費不少編譯時間。",
    ],
    diagram: ["巢狀迴圈", "迭代空間多面體", "相依分析", "求解新排程", "產生新迴圈"],
    code: `// Affine 形式的迴圈，才能被多面體工具分析
for (int i = 0; i < N; i++)
  for (int j = 0; j < i; j++)
    A[i][j] = A[i - 1][j] + 1;`,
    caseTitle: "Pluto 與 Polly",
    case: "Pluto 是多面體自動平行化的代表研究；LLVM 的 Polly 專案則把多面體最佳化帶進 LLVM。",
    related: ["loop-tiling", "mlir-affine", "tensor-comprehensions", "tiramisu"],
    sources: [["LLVM Polly", "https://polly.llvm.org/"]],
    papers: [p("A Practical Automatic Polyhedral Parallelizer and Locality Optimizer", "PLDI", 2008, "https://doi.org/10.1145/1375581.1375595", "Pluto 的原始論文。")],
  }),
  c({
    id: "loop-tiling",
    title: "Loop Tiling",
    zh: "迴圈分塊",
    level: "Core focus",
    summary: "把大迴圈切成多個小區塊處理，讓資料保持在快取中，以提高記憶體的使用效率。",
    intuition: "搬一大堆書時，不是整排來回走，而是一次處理一小疊，常用的東西都在手邊。",
    mechanism: [
      "把迭代空間切成 tile，每個 tile 都小到能放進快取。",
      "同一個 tile 內的資料會被重複利用，因而減少快取未命中。",
      "tile 大小要配合硬體，且轉換後必須遵守相依關係。",
    ],
    diagram: ["原始迴圈", "切成 tile", "逐 tile 計算", "提高快取命中"],
    code: `for (int ii = 0; ii < N; ii += T)
  for (int jj = 0; jj < N; jj += T)
    for (int i = ii; i < min(ii + T, N); i++)
      for (int j = jj; j < min(jj + T, N); j++)
        C[i][j] += A[i][j] * B[j][i];`,
    caseTitle: "線性代數與 ML 編譯器",
    case: "高效線性代數函式庫常用分塊技巧；MLIR 的 Linalg 也把 tiling 作為核心轉換之一。",
    related: ["linalg-dialect", "polyhedral-model", "loop-fusion", "gpu-memory"],
    sources: [["MLIR: Linalg Dialect", "https://mlir.llvm.org/docs/Dialects/Linalg/"]],
    papers: [p("A Data Locality Optimizing Algorithm", "PLDI", 1991, "https://doi.org/10.1145/113445.113449", "探討以分塊等轉換改善資料區域性。")],
  }),
  c({
    id: "auto-vectorization",
    title: "Auto-Vectorization",
    zh: "自動向量化",
    summary: "編譯器自動把純量運算改寫成 SIMD 指令，一次處理多筆資料。",
    intuition: "原本一次搬一箱，改用推車一次推八箱。前提是這八箱要往同一個方向。",
    mechanism: [
      "迴圈向量化把多次迭代合併成一條向量指令。",
      "SLP 向量化則把程式中相似的獨立運算打包在一起。",
      "別名、相依與對齊問題都可能阻止向量化，且需以成本模型評估是否值得。",
    ],
    diagram: ["純量迴圈", "相依與別名分析", "成本評估", "SIMD 指令"],
    code: `// clang -O3 -Rpass=loop-vectorize vec.c
void add(float *restrict a, float *restrict b, int n) {
  for (int i = 0; i < n; i++)
    a[i] += b[i];
}`,
    caseTitle: "LLVM 的向量化器",
    case: "LLVM 包含 Loop Vectorizer 和 SLP Vectorizer，文件並說明可用編譯選項查看向量化的結果。",
    related: ["llvm", "loop-tiling", "xnnpack", "onednn"],
    sources: [["LLVM: Auto-Vectorization", "https://llvm.org/docs/Vectorizers.html"]],
    papers: [p("Exploiting Superword Level Parallelism with Multimedia Instruction Sets", "PLDI", 2000, "https://doi.org/10.1145/349299.349320", "提出 SLP 向量化。")],
  }),
  c({
    id: "constant-folding",
    title: "Constant Folding",
    zh: "常數折疊",
    summary: "在編譯時就算出結果已知的運算，把它們換成常數，以節省執行時間。",
    intuition: "題目寫 3 × 4 就先算成 12，不必等考試時再算。",
    mechanism: [
      "找出所有輸入都已知的運算。",
      "在編譯時計算結果，並以常數取代。",
      "可以搭配常數傳播，連帶讓更多運算被折疊。ML 圖中也常用於折疊形狀計算與固定權重轉換。",
    ],
    diagram: ["找出常數輸入", "編譯時計算", "以常數取代", "移除多餘運算"],
    code: `# 使用 torch.fx 的工具進行常數折疊
from torch.fx.experimental.const_fold import split_const_subgraphs

folded = split_const_subgraphs(model)
folded.run_folding()`,
    caseTitle: "計算圖最佳化",
    case: "ONNX Runtime 的基本圖最佳化包含常數折疊，XLA 的 HLO 也有類似的 pass。",
    related: ["dead-code-elimination", "common-subexpression-elimination", "computation-graph", "onnx-runtime"],
    sources: [["ONNX Runtime: Graph optimizations", "https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html"]],
    papers: [p("Constant Propagation with Conditional Branches", "TOPLAS", 1991, "https://doi.org/10.1145/103135.103136", "常數傳播的經典演算法，常與常數折疊搭配使用。")],
  }),
  c({
    id: "dead-code-elimination",
    title: "Dead Code Elimination",
    zh: "死碼消除",
    summary: "移除結果不會被使用、也沒有副作用的運算，以減少計算量。",
    intuition: "整理工作流程時，把沒人會看的報表取消，省下做報表的時間。",
    mechanism: [
      "從輸出往回追蹤，標記會影響結果的運算。",
      "沒被標記、又沒有副作用的運算就可以移除。",
      "常和其他最佳化交替進行，因為它們會持續產生新的無用程式碼。",
    ],
    diagram: ["計算圖", "從輸出往回標記", "移除未標記節點", "精簡後的圖"],
    code: `import torch.fx as fx

gm = fx.symbolic_trace(model)
gm.graph.eliminate_dead_code()
gm.recompile()`,
    caseTitle: "torch.fx 的內建方法",
    case: "torch.fx.Graph 內建 eliminate_dead_code 方法；LLVM 也提供多種死碼消除 pass。",
    related: ["constant-folding", "common-subexpression-elimination", "torch-fx", "llvm-pass-manager"],
    sources: [LLVM_PASSES, ["PyTorch: torch.fx", "https://pytorch.org/docs/stable/fx.html"]],
    papers: [p("Partial Dead Code Elimination", "PLDI", 1994, "https://doi.org/10.1145/178243.178256", "把死碼消除延伸到只在部分路徑上無用的程式碼。")],
  }),
  c({
    id: "common-subexpression-elimination",
    title: "Common Subexpression Elimination",
    zh: "共同子運算式消除",
    summary: "找出重複計算的相同運算式，只算一次並重複使用結果。",
    intuition: "兩份報告都要同一個數字，算一次寫進共用表格就好，不必各算一遍。",
    mechanism: [
      "以雜湊或值編號判斷兩個運算是否等價。",
      "運算必須沒有副作用，且輸入相同。",
      "保留結果需要多佔記憶體，因此不一定總是划算。",
    ],
    diagram: ["計算圖", "找出等價節點", "合併節點", "重用結果"],
    code: `# 優化前
y = (a + b) * c
z = (a + b) * d
# 優化後
t = a + b
y = t * c
z = t * d`,
    caseTitle: "編譯器標準手法",
    case: "LLVM 的 GVN 與 EarlyCSE、ONNX Runtime 的圖最佳化，以及 AOTAutograd 的流程中，都有消除共同子運算式的步驟。",
    related: ["constant-folding", "dead-code-elimination", "equality-saturation", "llvm-pass-manager"],
    sources: [LLVM_PASSES],
    papers: [p("Global Code Motion / Global Value Numbering", "PLDI", 1995, "https://doi.org/10.1145/207110.207154", "以全域值編號找出等價運算。")],
  }),
  c({
    id: "equality-saturation",
    title: "Equality Saturation",
    zh: "等式飽和",
    summary: "把所有可能的改寫結果同時存進 e-graph，最後再挑出成本最低的版本，避開改寫順序的問題。",
    intuition: "不必決定先走哪條路，而是把所有路線都畫在地圖上，再選最短的一條。",
    mechanism: [
      "傳統改寫依序進行，順序不同，結果可能不同。",
      "E-graph 能緊湊表示多個等價的運算式。",
      "持續套用規則直到飽和或達到限制，再依成本函數取出結果。",
    ],
    diagram: ["原始運算式", "建立 E-graph", "套用改寫規則", "飽和", "取出最低成本"],
    code: `// 使用 Rust 的 egg 函式庫
let rules: &[Rewrite<SymbolLang, ()>] = &[
    rewrite!("commute"; "(* ?a ?b)" => "(* ?b ?a)"),
    rewrite!("div-one"; "(/ ?a 1)" => "?a"),
];`,
    caseTitle: "egg 函式庫",
    case: "egg 是開源的 e-graph 函式庫，已用於多種程式語言與編譯器的研究。",
    related: ["graph-superoptimization", "common-subexpression-elimination", "peephole-optimization", "computation-graph"],
    sources: [["egg (GitHub)", "https://github.com/egraphs-good/egg"]],
    papers: [p("egg: Fast and Extensible Equality Saturation", "POPL", 2021, "https://arxiv.org/abs/2004.03082", "提出高效且可擴充的 e-graph 實作。")],
  }),
  c({
    id: "graph-superoptimization",
    title: "Graph Superoptimization",
    zh: "計算圖超級最佳化",
    summary: "自動產生並驗證圖改寫規則，再搜尋成本較低的計算圖，以減少人工撰寫規則的負擔。",
    intuition: "不靠老師傅的固定口訣，而是用電腦自己試出更省料的做法，並且驗證它確實正確。",
    mechanism: [
      "列舉小型子圖，找出數學上等價的組合並產生規則。",
      "以形式化方法驗證規則的正確性。",
      "以成本模型引導搜尋，將規則組合應用到實際模型上。",
    ],
    diagram: ["列舉運算子組合", "驗證等價", "產生改寫規則", "成本引導搜尋", "較佳計算圖"],
    code: `# 示意：以回溯搜尋套用規則
best = graph
queue = [graph]
while queue and budget_left():
    g = queue.pop(0)
    for rule in verified_rules:
        for new_g in rule.apply(g):
            if cost(new_g) < cost(best) * alpha:
                queue.append(new_g)
                best = min(best, new_g, key=cost)`,
    caseTitle: "TASO",
    case: "TASO 自動產生並驗證改寫規則，並以此最佳化 TensorFlow 計算圖。",
    related: ["equality-saturation", "learned-cost-model", "operator-fusion", "computation-graph"],
    sources: [["TASO (GitHub)", "https://github.com/jiazhihao/TASO"]],
    papers: [p("TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions", "SOSP", 2019, "https://doi.org/10.1145/3341301.3359630", "自動產生並驗證圖改寫規則。")],
  }),
  c({
    id: "dynamic-shape-compilation",
    title: "Dynamic Shape Compilation",
    zh: "動態形狀編譯",
    summary: "讓編譯後的程式能處理執行時才確定的張量形狀，以避免每個新形狀都要重新編譯。",
    intuition: "與其替每種尺寸各開一條生產線，不如做一條可調整尺寸的生產線。",
    mechanism: [
      "許多編譯器針對固定形狀特化，形狀改變就要重新編譯。",
      "動態形狀編譯以符號表示維度，並保留維度之間的約束。",
      "也有折衷做法，例如把輸入補齊到幾種固定大小（bucketing）。",
    ],
    diagram: ["符號形狀", "形狀約束分析", "編譯可處理多形狀的程式", "執行時才確定維度"],
    code: `import torch
from torch.export import Dim, export

batch = Dim("batch")
ep = export(model, (x,), dynamic_shapes={"x": {0: batch}})`,
    caseTitle: "LLM 的可變長度",
    case: "LLM 的 prompt 長度變化很大，因此形狀處理是編譯時必須考慮的問題。DISC 研究以 MLIR 建立動態形狀編譯流程。",
    related: ["xla", "torch-compile", "mlir", "static-memory-planning"],
    sources: [["PyTorch: torch.export", "https://pytorch.org/docs/stable/export.html"]],
    papers: [p("DISC: A Dynamic Shape Compiler for Machine Learning Workloads", "EuroMLSys", 2021, "https://arxiv.org/abs/2103.05288", "以 MLIR 建立動態形狀編譯流程。")],
  }),
  c({
    id: "static-memory-planning",
    title: "Static Memory Planning",
    zh: "靜態記憶體規劃",
    summary: "在編譯時分析張量的生命週期，讓不重疊的張量共用記憶體，以降低推論的記憶體尖峰。",
    intuition: "會議室排程：兩場不同時段的會議可以共用同一間，不必各訂一間。",
    mechanism: [
      "依計算圖算出每個張量何時產生、何時不再使用。",
      "把記憶體配置視為排程問題，常用貪婪演算法處理。",
      "形狀需要事先知道或有上界，動態形狀會增加難度。",
    ],
    diagram: ["計算圖", "分析張量生命週期", "分配位移", "共用緩衝區"],
    code: `# 示意：貪婪配置
for tensor in sorted(tensors, key=lambda t: -t.size):
    offset = first_fit(tensor, overlapping(tensor, placed))
    tensor.offset = offset
    placed.append(tensor)`,
    caseTitle: "TensorFlow Lite",
    case: "相關研究出自 TensorFlow Lite 團隊，探討裝置端推論時，如何以簡單的策略降低記憶體使用量。",
    related: ["bufferization", "glow", "executorch", "register-allocation"],
    sources: [["Efficient Memory Management for DNN Inference (arXiv)", "https://arxiv.org/abs/2001.03288"]],
    papers: [p("Efficient Memory Management for Deep Neural Net Inference", "arXiv", 2020, "https://arxiv.org/abs/2001.03288", "比較裝置端推論的記憶體配置策略。")],
  }),
  c({
    id: "bufferization",
    title: "Bufferization",
    zh: "緩衝區化",
    summary: "把 MLIR 中不可變的張量值轉成實際的記憶體緩衝區，並盡量重用以避免多餘的複製。",
    intuition: "設計圖上只寫「產生新零件」；真正施工時才要決定放在哪個架子、能不能直接改舊零件。",
    mechanism: [
      "tensor 型別是值語意，memref 則對應實際記憶體。",
      "one-shot bufferize 分析讀寫衝突，判斷能否就地更新。",
      "無法就地時才插入複製，以確保語意不變。",
    ],
    diagram: ["tensor 值語意", "分析讀寫衝突", "就地或複製", "memref 緩衝區"],
    code: `mlir-opt input.mlir \\
  --one-shot-bufferize="bufferize-function-boundaries" \\
  -o output.mlir`,
    caseTitle: "MLIR 的 one-shot bufferize",
    case: "MLIR 文件說明了 one-shot bufferize 的分析方式，以及如何盡量減少複製。",
    related: ["mlir", "static-memory-planning", "linalg-dialect", "register-allocation"],
    sources: [["MLIR: Bufferization", "https://mlir.llvm.org/docs/Bufferization/"]],
    papers: [MLIR_PAPER("說明 MLIR 的多層次設計，bufferization 是其中的降階步驟。")],
  }),
  c({
    id: "register-allocation",
    title: "Register Allocation",
    zh: "暫存器配置",
    summary: "決定把變數放到有限的暫存器，還是記憶體中，對程式效能影響很大。",
    intuition: "手邊的工作檯位置有限，把最常用的工具留在檯面上，其他的收進櫃子裡。",
    mechanism: [
      "分析變數的存活區間，存活重疊的變數不能共用暫存器。",
      "圖著色法品質好，但較慢；線性掃描則較快。",
      "放不下時就 spill 到記憶體，會增加存取成本。",
    ],
    diagram: ["存活區間分析", "建構衝突", "配置暫存器", "必要時 spill"],
    code: `# 示意：線性掃描
for interval in sorted(intervals, key=lambda i: i.start):
    expire_old(active, interval.start)
    if free_registers:
        interval.reg = free_registers.pop()
        active.add(interval)
    else:
        spill(interval)`,
    caseTitle: "JIT 與 GPU 後端",
    case: "線性掃描因速度快，常用於 JIT 編譯器；GPU 程式也要留意暫存器用量，因為它會影響可同時執行的執行緒數。",
    related: ["llvm", "static-memory-planning", "tracing-jit", "nvidia-ptx"],
    sources: [["LLVM: Code Generator", "https://llvm.org/docs/CodeGenerator.html"]],
    papers: [p("Linear Scan Register Allocation", "TOPLAS", 1999, "https://doi.org/10.1145/330249.330250", "提出快速的線性掃描配置法。")],
  }),
  c({
    id: "halide",
    title: "Halide",
    zh: "Halide 影像處理語言",
    summary: "把演算法與排程分開描述的領域專用語言，影響了後來許多機器學習編譯器的設計。",
    intuition: "食譜寫「要做什麼」，另外一張工作表寫「誰在什麼時候做」。換工作方式，不必改食譜。",
    mechanism: [
      "演算法以純函數描述每個點的計算方式。",
      "排程另外決定分塊、向量化、平行化和存放方式。",
      "可以針對不同的硬體更換排程，並可自動搜尋。",
    ],
    diagram: ["演算法", "排程", "Halide 編譯器", "最佳化後的程式"],
    code: `Func blur_x, blur_y;
Var x, y, xi, yi;
blur_x(x, y) = (in(x-1, y) + in(x, y) + in(x+1, y)) / 3;
blur_y(x, y) = (blur_x(x, y-1) + blur_x(x, y) + blur_x(x, y+1)) / 3;
blur_y.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);`,
    caseTitle: "演算法與排程分離",
    case: "TVM 的論文提到其張量表示與排程概念受到 Halide 的影響，這種分離也成了許多張量編譯器的基本想法。",
    related: ["apache-tvm", "ansor", "loop-tiling", "tiramisu"],
    sources: [["Halide", "https://halide-lang.org/"]],
    papers: [p("Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines", "PLDI", 2013, "https://doi.org/10.1145/2491956.2462176", "Halide 的原始論文。")],
  }),
  c({
    id: "taco-compiler",
    title: "TACO",
    zh: "TACO 張量代數編譯器",
    summary: "把張量代數運算式與資料格式分開描述，自動產生適用於稀疏或密集資料的程式。",
    intuition: "公式是一回事，資料怎麼存是另一回事。給定兩者，編譯器幫你寫好對應的程式。",
    mechanism: [
      "使用者以索引記號描述運算式，例如 a(i) = B(i,j) * c(j)。",
      "為每個維度指定密集或壓縮等格式。",
      "編譯器產生合併迭代的程式，以有效走訪非零元素。",
    ],
    diagram: ["張量運算式", "格式宣告", "迭代圖", "產生 C 程式"],
    code: `Format csr({Dense, Sparse});
Tensor<double> A({n, n}, csr), x({n}, Format{Dense}), y({n}, Format{Dense});
IndexVar i, j;
y(i) = A(i, j) * x(j);
y.compile(); y.assemble(); y.compute();`,
    caseTitle: "稀疏張量編譯",
    case: "TACO 的概念影響了後來的稀疏編譯研究，包括 MLIR 的 Sparsifier。",
    related: ["sparse-tensor-compiler", "linalg-dialect", "halide", "mlir"],
    sources: [["TACO", "http://tensor-compiler.org/"]],
    papers: [p("The Tensor Algebra Compiler", "OOPSLA", 2017, "https://doi.org/10.1145/3133901", "TACO 的原始論文。")],
  }),
  c({
    id: "ansor",
    title: "Ansor",
    zh: "Ansor 自動排程",
    summary: "TVM 的自動排程系統，自動產生張量程式並以進化搜尋與學習模型找出較好的程式。",
    intuition: "不請人手動排工作表，而是讓系統試很多種方案，再用經驗模型挑出最有希望的。",
    mechanism: [
      "以規則自動產生大量候選的張量程式。",
      "以隨機抽樣與進化搜尋微調參數。",
      "以學習的成本模型挑選值得實測的候選程式。",
    ],
    diagram: ["計算子圖", "規則產生候選", "進化搜尋", "學習成本模型", "實測挑選"],
    code: `# 舊版 TVM 介面示意；新版請參考 Relax 文件
from tvm.auto_scheduler import TuningOptions

options = TuningOptions(num_measure_trials=200)
# 交由 tvm.driver.tune 啟動自動排程`,
    caseTitle: "TVM 的自動調校",
    case: "Ansor 由 TVM 社群整合為 auto-scheduler，並有論文說明其設計。",
    related: ["apache-tvm", "learned-cost-model", "halide", "mlgo"],
    sources: [["Apache TVM documentation", "https://tvm.apache.org/docs/"]],
    papers: [p("Ansor: Generating High-Performance Tensor Programs for Deep Learning", "OSDI", 2020, "https://arxiv.org/abs/2006.06762", "Ansor 的原始論文。")],
  }),
  c({
    id: "learned-cost-model",
    title: "Learned Cost Model",
    zh: "學習型成本模型",
    summary: "以機器學習預測程式的執行時間，讓編譯器不必每次都實際執行，就能比較多個最佳化方案。",
    intuition: "資深主廚看一眼菜單就知道大概多久能出餐，不必每次都實際煮一遍。",
    mechanism: [
      "傳統成本模型靠人工規則，硬體一複雜就難以準確。",
      "蒐集大量程式與實測時間，訓練圖神經網路等模型。",
      "編譯器以預測結果來引導搜尋，並只實測少數候選。",
    ],
    diagram: ["程式資料與實測時間", "訓練預測模型", "預測候選方案", "挑選並實測"],
    code: `# 示意
features = encode_graph(program)
predicted_ms = cost_model(features)
best = min(candidates, key=lambda p: cost_model(encode_graph(p)))`,
    caseTitle: "TPU 的效能模型",
    case: "Google 研究以圖神經網路預測 TPU 程式的執行時間，並評估將其用於編譯器的自動調校。",
    related: ["ansor", "apache-tvm", "mlgo", "graph-superoptimization"],
    sources: [["A Learned Performance Model for TPUs (arXiv)", "https://arxiv.org/abs/2008.01040"]],
    papers: [p("A Learned Performance Model for Tensor Processing Units", "MLSys", 2021, "https://arxiv.org/abs/2008.01040", "以圖神經網路預測 TPU 程式效能。")],
  }),
  c({
    id: "mlgo",
    title: "MLGO",
    zh: "MLGO 機器學習引導最佳化",
    summary: "把 LLVM 的啟發式規則換成訓練好的模型，例如決定是否函式內聯，以及暫存器如何驅逐。",
    intuition: "把編譯器裡寫死的經驗法則，換成看過大量案例後學出來的判斷。",
    mechanism: [
      "編譯器輸出每次決策的特徵。",
      "以強化學習或模仿學習訓練模型。",
      "把模型編譯進 LLVM，不必在編譯時依賴完整的機器學習環境。",
    ],
    diagram: ["LLVM 決策點", "蒐集特徵", "訓練策略模型", "嵌入編譯器"],
    code: `# 需要以支援 MLGO 的方式建置 LLVM
clang -O2 -mllvm -enable-ml-inliner=release file.c -c`,
    caseTitle: "Google 的 LLVM 研究",
    case: "Google 開源 ml-compiler-opt 專案，並展示以 MLGO 調整函式內聯來縮小程式大小。",
    related: ["llvm", "learned-cost-model", "ansor", "bolt-optimizer"],
    sources: [["ml-compiler-opt (GitHub)", "https://github.com/google/ml-compiler-opt"]],
    papers: [p("MLGO: a Machine Learning Guided Compiler Optimizations Framework", "arXiv", 2021, "https://arxiv.org/abs/2101.04808", "說明如何把機器學習整合進 LLVM。")],
  }),
  c({
    id: "systolic-array",
    title: "Systolic Array",
    zh: "脈動陣列",
    summary: "由大量運算單元組成的網格，資料在單元間規律流動，非常適合矩陣相乘。",
    intuition: "像工廠傳送帶，每個工人處理後就往下傳，不必每次都跑回倉庫拿材料。",
    mechanism: [
      "每個 PE 執行乘加運算，然後把資料傳給相鄰的 PE。",
      "資料可在陣列內重複使用，減少記憶體存取。",
      "編譯器要把運算切成符合陣列大小的區塊，並安排資料移動。",
    ],
    diagram: ["權重載入", "資料流過 PE 網格", "累加結果", "輸出緩衝區"],
    code: `# 示意：矩陣相乘切塊以符合陣列大小
for i in range(0, M, ARRAY_DIM):
    for j in range(0, N, ARRAY_DIM):
        run_on_array(A[i:i + ARRAY_DIM], B[:, j:j + ARRAY_DIM])`,
    caseTitle: "Google TPU",
    case: "Google 公開的第一代 TPU 論文，說明了其以 256×256 的脈動陣列處理矩陣相乘。",
    related: ["gemmini", "loop-tiling", "xla", "tensor-cores"],
    sources: [["Google Cloud: TPU architecture", "https://cloud.google.com/tpu/docs/system-architecture-tpu-vm"]],
    papers: [p("In-Datacenter Performance Analysis of a Tensor Processing Unit", "ISCA", 2017, "https://arxiv.org/abs/1704.04760", "第一代 TPU 的設計與資料中心評估。")],
  }),
  c({
    id: "gemmini",
    title: "Gemmini",
    zh: "Gemmini 加速器產生器",
    summary: "開源的深度學習加速器產生器，可用於研究軟硬體堆疊對效能的影響。",
    intuition: "像可以自己設定零件的樂高工廠，能產生不同規格的加速器，並放到完整系統中測試。",
    mechanism: [
      "可設定脈動陣列大小、暫存空間與資料型別。",
      "與 RISC-V SoC 整合，能在完整系統上評估。",
      "提供軟體堆疊，讓研究者觀察系統層效應。",
    ],
    diagram: ["參數設定", "產生硬體", "與 RISC-V SoC 整合", "軟體堆疊評估"],
    code: `// Gemmini 以 Chisel 參數設定
// 例如 meshRows、meshColumns 控制陣列大小
// 細節請參考 ucb-bar/gemmini 的說明文件`,
    caseTitle: "學術研究平台",
    case: "UC Berkeley 的 Gemmini 被用來研究加速器設計，論文強調系統層面的因素也會明顯影響效能。",
    related: ["systolic-array", "apache-tvm", "onednn", "quantization"],
    sources: [["Gemmini (GitHub)", "https://github.com/ucb-bar/gemmini"]],
    papers: [p("Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration", "DAC", 2021, "https://arxiv.org/abs/1911.09925", "Gemmini 的原始論文。")],
  }),
  c({
    id: "device-placement",
    title: "Device Placement",
    zh: "裝置放置最佳化",
    summary: "決定計算圖中的每個運算要放在哪個裝置上，並兼顧計算時間與資料傳輸成本。",
    intuition: "分配工作給不同部門時，不只看誰做得快，也要考慮部門之間傳文件要花多久。",
    mechanism: [
      "把運算分配到 CPU、GPU 或加速器上。",
      "跨裝置的邊會產生傳輸成本，因此切分太細可能反而更慢。",
      "研究中有以啟發式方法或強化學習進行搜尋，實務上則常使用規則或框架預設。",
    ],
    diagram: ["計算圖", "切分子圖", "分配裝置", "評估計算與傳輸", "最終放置"],
    code: `import onnxruntime as ort

# ONNX Runtime 依序把子圖分配給可用的 execution provider
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)`,
    caseTitle: "ONNX Runtime 的子圖分配",
    case: "ONNX Runtime 依 execution provider 的順序，把可支援的子圖分配出去，其餘則退回 CPU。",
    related: ["onnx-runtime", "computation-graph", "learned-cost-model", "tensorrt"],
    sources: [["ONNX Runtime: Execution Providers", "https://onnxruntime.ai/docs/execution-providers/"]],
    papers: [p("Device Placement Optimization with Reinforcement Learning", "ICML", 2017, "https://arxiv.org/abs/1706.04972", "以強化學習搜尋放置方式的早期研究。")],
  }),
  c({
    id: "tracing-jit",
    title: "Tracing JIT",
    zh: "追蹤式即時編譯",
    summary: "在執行時記錄實際走過的路徑，再把這些常用路徑編譯成快速的程式。",
    intuition: "觀察員工每天實際走的路線，只把最常走的那條路鋪好。",
    mechanism: [
      "先以直譯方式執行，並偵測常執行的程式碼。",
      "記錄實際執行的操作序列，形成追蹤（trace）。",
      "以 guard 檢查假設，一旦路徑不同就退回直譯。",
    ],
    diagram: ["直譯執行", "偵測熱點", "記錄 trace", "編譯", "guard 失敗則退回"],
    code: `import torch

traced = torch.jit.trace(model, example)
# 注意：trace 只記錄這次走的路徑
# 依輸入值變化的 Python if 不會被保留`,
    caseTitle: "與機器學習框架的關聯",
    case: "TorchDynamo 同樣以 guard 檢查追蹤時的假設；torch.jit.trace 則是常見的追蹤式匯出，並會提示控制流可能被固定。",
    related: ["torch-dynamo", "torchscript", "torch-fx", "register-allocation"],
    sources: [["PyTorch: TorchScript", "https://pytorch.org/docs/stable/jit.html"]],
    papers: [p("Trace-based Just-in-Time Type Specialization for Dynamic Languages", "PLDI", 2009, "https://doi.org/10.1145/1542476.1542528", "TraceMonkey，動態語言追蹤式 JIT 的代表研究。")],
  }),
  c({
    id: "enzyme-ad",
    title: "Enzyme",
    zh: "Enzyme 編譯器自動微分",
    summary: "在 LLVM IR 上進行自動微分，讓最佳化之後的程式也能被微分，不限特定語言。",
    intuition: "不是在每種語言裡各做一次微分，而是在大家都會翻成的那份中介語言上做一次。",
    mechanism: [
      "在 LLVM IR 上分析活躍變數，判斷哪些值會影響導數。",
      "產生反向模式或前向模式的梯度函式。",
      "在最佳化之後才微分，能受益於先前的簡化結果。",
    ],
    diagram: ["原始程式", "LLVM IR 最佳化", "Enzyme 微分", "梯度函式", "再次最佳化"],
    code: `// 使用 Enzyme 的 C 介面
double square(double x) { return x * x; }
extern double __enzyme_autodiff(void*, double);

double grad(double x) {
  return __enzyme_autodiff((void*)square, x);
}`,
    caseTitle: "跨語言自動微分",
    case: "Enzyme 可用於多種能編譯到 LLVM 的語言，例如 C/C++、Julia 與 Rust。",
    related: ["autograd", "llvm", "aot-autograd", "backpropagation"],
    sources: [["Enzyme", "https://enzyme.mit.edu/"]],
    papers: [p("Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients", "NeurIPS", 2020, "https://arxiv.org/abs/2010.01709", "Enzyme 的原始論文。")],
  }),
  c({
    id: "spir-v",
    title: "SPIR-V",
    zh: "SPIR-V 中間表示",
    summary: "Khronos 定義的二進位中間表示，用於 Vulkan 與 OpenCL 等異質運算環境。",
    intuition: "GPU 世界的通用貨櫃格式，不同語言都可以裝進來，再由各家驅動程式接手。",
    mechanism: [
      "以 SSA 形式表示著色器與運算 kernel。",
      "前端把 GLSL、HLSL 或 OpenCL C 編譯成 SPIR-V。",
      "驅動程式再把它轉成特定 GPU 的機器碼。",
    ],
    diagram: ["GLSL / OpenCL C", "SPIR-V 二進位", "GPU 驅動程式", "機器碼"],
    code: `# 把 GLSL 運算著色器編譯成 SPIR-V 並反組譯
glslangValidator -V compute.comp -o compute.spv
spirv-dis compute.spv`,
    caseTitle: "跨廠商 GPU 運算",
    case: "SPIR-V 是 Vulkan 所要求的著色器格式；MLIR 也提供 SPIR-V 方言。",
    related: ["nvidia-ptx", "mlir", "iree", "llvm"],
    sources: [["Khronos: SPIR-V", "https://www.khronos.org/spirv/"], ["MLIR: SPIR-V Dialect", "https://mlir.llvm.org/docs/Dialects/SPIR-V/"]],
  }),
  c({
    id: "nvidia-ptx",
    title: "NVIDIA PTX",
    zh: "PTX 平行執行緒虛擬指令集",
    summary: "NVIDIA GPU 的虛擬指令集，由 CUDA 編譯器產生，再轉成特定 GPU 的機器碼（SASS）。",
    intuition: "寫給一般 NVIDIA GPU 看的通用指令，到了具體型號，再翻成它專用的語言。",
    mechanism: [
      "CUDA 程式先被編譯成 PTX。",
      "ptxas 或驅動程式的 JIT 把 PTX 轉成 SASS。",
      "PTX 使用虛擬暫存器，實際配置則發生在轉換成 SASS 時。",
    ],
    diagram: ["CUDA C++", "LLVM / nvcc", "PTX", "ptxas 或 JIT", "SASS"],
    code: `# 取得 Triton kernel 編譯後的 PTX
compiled = add_kernel.warmup(a, b, c, N, grid=(N,))
print(compiled.asm["ptx"])

# 反組譯 CUDA 物件檔中的 SASS（於 shell 執行）
# cuobjdump -sass kernel.o`,
    caseTitle: "Triton 的編譯路徑",
    case: "Triton 在 NVIDIA 平台上會經由 MLIR 與 LLVM 產生 PTX，再轉換成機器碼。",
    related: ["cuda", "triton", "spir-v", "register-allocation"],
    sources: [["NVIDIA: PTX ISA", "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"]],
  }),
  c({
    id: "xnnpack",
    title: "XNNPACK",
    zh: "XNNPACK 推論函式庫",
    summary: "Google 開源的高效能推論運算子函式庫，主要針對 ARM、x86 等平台的 CPU。",
    intuition: "為不同的 CPU 分別準備手工調校的工具組，框架只要呼叫，就能得到不錯的效能。",
    mechanism: [
      "針對各種指令集（如 NEON、AVX）提供最佳化的 micro-kernel。",
      "以工作切分和多執行緒處理資料。",
      "TensorFlow Lite 與 ExecuTorch 都可以把運算委派給它。",
    ],
    diagram: ["框架委派子圖", "XNNPACK 運算子", "選擇 micro-kernel", "多執行緒執行"],
    code: `# ExecuTorch 委派給 XNNPACK
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge

edge = to_edge(ep).to_backend(XnnpackPartitioner())
program = edge.to_executorch()`,
    caseTitle: "行動裝置後端",
    case: "XNNPACK 是 ExecuTorch 文件列出的 CPU 後端之一，也被其他框架用於 CPU 推論。",
    related: ["executorch", "auto-vectorization", "onednn", "quantization"],
    sources: [["XNNPACK (GitHub)", "https://github.com/google/XNNPACK"]],
  }),
  c({
    id: "onednn",
    title: "oneDNN",
    zh: "oneDNN 深度學習函式庫",
    summary: "UXL Foundation 下的開源函式庫，提供最佳化的深度學習基本運算，以支援 CPU 與 GPU。",
    intuition: "替 CPU 量身打造的矩陣工具箱，框架不必自己把每個指令都調校一次。",
    mechanism: [
      "以執行時程式碼生成，依 CPU 能力選擇最佳的實作。",
      "提供 primitive 與 graph 兩種介面，後者可融合多個運算。",
      "會使用特定的記憶體布局來提高效率。",
    ],
    diagram: ["框架呼叫", "偵測 CPU 能力", "JIT 產生 kernel", "以最佳布局執行"],
    code: `import torch

# PyTorch 在 x86 上通常會使用 oneDNN（MKLDNN）
print(torch.backends.mkldnn.is_available())
# 設定 ONEDNN_VERBOSE=1 可以印出實際使用的 kernel`,
    caseTitle: "框架的 CPU 後端",
    case: "PyTorch 與 TensorFlow 都能使用 oneDNN 來最佳化 CPU 上的運算。",
    related: ["xnnpack", "openvino", "auto-vectorization", "operator-fusion"],
    sources: [["oneDNN (GitHub)", "https://github.com/uxlfoundation/oneDNN"]],
  }),
  c({
    id: "openvino",
    title: "OpenVINO",
    zh: "OpenVINO 工具組",
    summary: "Intel 主導的開源推論工具組，可轉換模型，並在 CPU、GPU 與 NPU 上最佳化執行。",
    intuition: "讓同一份模型能在 Intel 不同硬體上跑得好的轉接與調校工具。",
    mechanism: [
      "把 PyTorch、ONNX 等模型轉成 OpenVINO 的 IR。",
      "外掛依目標裝置進行圖最佳化與編譯。",
      "提供 AUTO 裝置模式與量化工具。",
    ],
    diagram: ["PyTorch / ONNX", "轉成 OpenVINO IR", "選擇裝置外掛", "編譯", "推論"],
    code: `import openvino as ov

core = ov.Core()
model = core.read_model("model.onnx")
compiled = core.compile_model(model, "AUTO")
result = compiled(inputs)`,
    caseTitle: "Intel 硬體上的推論",
    case: "OpenVINO 支援 Intel CPU、GPU 與 NPU，也有與 Hugging Face 整合的 Optimum Intel。",
    related: ["onednn", "onnx", "quantization", "tensorrt"],
    sources: [["OpenVINO documentation", "https://docs.openvino.ai/"]],
  }),
  c({
    id: "mlir-affine",
    title: "MLIR Affine Dialect",
    zh: "MLIR Affine 方言",
    summary: "以仿射運算式描述迴圈與記憶體存取的方言，便於精確分析與多面體風格的轉換。",
    intuition: "規定迴圈與索引只能寫成簡單的線性形式，編譯器因此能精確算出誰會碰到誰。",
    mechanism: [
      "affine.for 的邊界與存取索引必須是仿射運算式。",
      "精確的相依分析，讓分塊與融合等轉換更安全。",
      "最後再降階到 scf 與 LLVM 方言。",
    ],
    diagram: ["Linalg / 高階運算", "Affine 迴圈", "相依分析與轉換", "scf", "LLVM"],
    code: `affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %v = affine.load %A[%i, %j] : memref<100x100xf32>
    affine.store %v, %B[%j, %i] : memref<100x100xf32>
  }
}`,
    caseTitle: "MLIR 的迴圈最佳化",
    case: "MLIR 文件提供 Affine 方言的規格，並附上相關的迴圈轉換 pass。",
    related: ["mlir", "polyhedral-model", "linalg-dialect", "loop-tiling"],
    sources: [["MLIR: Affine Dialect", "https://mlir.llvm.org/docs/Dialects/Affine/"]],
    papers: [MLIR_PAPER("說明 MLIR 如何表示多面體相關的資訊。")],
  }),
  c({
    id: "sparse-tensor-compiler",
    title: "Sparse Tensor Compilation",
    zh: "稀疏張量編譯",
    summary: "把演算法與稀疏資料格式分開，由編譯器自動產生走訪非零元素的程式。",
    intuition: "寫程式時當作資料都是滿的，編譯器會自動跳過空格子，只處理有內容的地方。",
    mechanism: [
      "以型別屬性標註張量的稀疏格式。",
      "編譯器把運算合併迭代，並只走訪非零元素。",
      "同一段演算法，可以替換不同的稀疏格式來比較效能。",
    ],
    diagram: ["以密集形式撰寫", "標註稀疏格式", "Sparsifier", "稀疏迴圈"],
    code: `#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>
// 把此編碼加到張量型別上：tensor<?x?xf64, #CSR>`,
    caseTitle: "MLIR 的 Sparsifier",
    case: "MLIR 提供 SparseTensor 方言，相關論文說明如何把稀疏性視為型別屬性，並自動產生程式。",
    related: ["taco-compiler", "linalg-dialect", "mlir", "weight-pruning"],
    sources: [["MLIR: SparseTensor Dialect", "https://mlir.llvm.org/docs/Dialects/SparseTensorOps/"]],
    papers: [p("Compiler Support for Sparse Tensor Computations in MLIR", "TACO", 2022, "https://arxiv.org/abs/2202.04305", "說明 MLIR 中的稀疏編譯支援。")],
  }),
  c({
    id: "peephole-optimization",
    title: "Peephole Optimization",
    zh: "窺孔最佳化",
    summary: "只觀察一小段相鄰指令，把它們換成更快或更短的等價寫法。",
    intuition: "校稿時只看相鄰的幾個字，把「每一個每一個」改成「每一個」。範圍小，但效果累積起來不少。",
    mechanism: [
      "比對一小段指令或子圖的模式。",
      "把它換成等價但更好的形式，例如移除相反的轉置運算。",
      "規則簡單快速，常會重複套用到不再有變化為止。",
    ],
    diagram: ["滑動視窗", "比對規則", "替換", "重複直到穩定"],
    code: `# 示意：移除連續兩次、互相抵銷的 transpose
def simplify(node):
    if (node.op == "transpose" and node.input.op == "transpose"
            and compose(node.perm, node.input.perm) == identity):
        return node.input.input
    return node`,
    caseTitle: "計算圖的代數化簡",
    case: "XLA 的 AlgebraicSimplifier 與 PyTorch 的圖最佳化都會應用類似的局部改寫。",
    related: ["constant-folding", "strength-reduction", "equality-saturation", "operator-fusion"],
    sources: [["OpenXLA: XLA", "https://openxla.org/xla"]],
    papers: [p("Peephole Optimization", "CACM", 1965, "https://doi.org/10.1145/364995.365000", "最早提出這個技巧的論文。")],
  }),
  c({
    id: "bolt-optimizer",
    title: "BOLT",
    zh: "BOLT 二進位最佳化器",
    summary: "在連結後依實際執行資料重新排列程式碼，以改善指令快取，並提高大型程式的效能。",
    intuition: "蓋好房子後觀察動線，把最常一起用的東西擺在一起，不必重新蓋房子。",
    mechanism: [
      "以 perf 等工具收集程式實際執行的資料。",
      "分析二進位，並依資料重排基本區塊與函式。",
      "常與 LTO、PGO 互補，用來進一步改善程式配置。",
    ],
    diagram: ["連結後的二進位", "收集 perf 資料", "BOLT 重排", "最佳化後的二進位"],
    code: `perf record -e cycles:u -j any,u -o perf.data -- ./app
perf2bolt -p perf.data -o perf.fdata ./app
llvm-bolt ./app -o ./app.bolt -data=perf.fdata \\
  -reorder-blocks=ext-tsp -reorder-functions=hfsort`,
    caseTitle: "Meta 的資料中心應用",
    case: "BOLT 由 Facebook（現為 Meta）開發，用於大型資料中心應用，現在已成為 LLVM 專案的一部分。",
    related: ["thinlto", "llvm", "mlgo", "auto-vectorization"],
    sources: [["LLVM BOLT (GitHub)", "https://github.com/llvm/llvm-project/tree/main/bolt"]],
    papers: [p("BOLT: A Practical Binary Optimizer for Data Centers and Beyond", "CGO", 2019, "https://arxiv.org/abs/1807.06735", "BOLT 的原始論文。")],
  }),
  c({
    id: "thinlto",
    title: "ThinLTO",
    zh: "ThinLTO 連結時最佳化",
    summary: "可擴充的連結時最佳化，以模組摘要做跨檔案分析，兼顧編譯速度與最佳化效果。",
    intuition: "不必把所有部門的文件攤在一起，只看每個部門的摘要，就能決定哪些工作要合併。",
    mechanism: [
      "各模組先產生 IR 與摘要。",
      "連結階段進行全域分析，並決定跨模組的匯入與內聯。",
      "之後各模組仍可平行處理，因此比傳統 LTO 容易擴充。",
    ],
    diagram: ["各模組 IR 與摘要", "全域索引", "平行最佳化", "連結"],
    code: `clang -flto=thin -O2 -c a.cpp b.cpp
clang -flto=thin -O2 a.o b.o -o app`,
    caseTitle: "大型 C++ 專案",
    case: "PyTorch、XLA 等專案含有大量 C++ 程式碼；LLVM 文件說明了以 ThinLTO 進行建置的方式。",
    related: ["bolt-optimizer", "llvm", "cpp", "llvm-pass-manager"],
    sources: [["LLVM: ThinLTO", "https://clang.llvm.org/docs/ThinLTO.html"]],
    papers: [p("ThinLTO: Scalable and Incremental LTO", "CGO", 2017, "https://doi.org/10.1109/CGO.2017.7863733", "ThinLTO 的原始論文。")],
  }),
  c({
    id: "loop-fusion",
    title: "Loop Fusion",
    zh: "迴圈融合",
    summary: "把走訪相同範圍的迴圈合併，以減少暫存資料並提高快取區域性。",
    intuition: "原本巡兩次走廊，一次關燈、一次鎖門；合併成一趟，邊走邊做，就省下許多路。",
    mechanism: [
      "檢查迴圈的範圍是否相容，以及合併後是否違反相依關係。",
      "合併後，可能不必再寫出中間陣列。",
      "operator fusion 在張量層做類似的事，例如把逐元素運算合併到同一個 kernel 中。",
    ],
    diagram: ["迴圈 A", "迴圈 B", "相依檢查", "合併迴圈", "移除中間陣列"],
    code: `// 融合前
for (int i = 0; i < n; i++) b[i] = a[i] * 2;
for (int i = 0; i < n; i++) c[i] = b[i] + 1;
// 融合後，可以不再需要 b
for (int i = 0; i < n; i++) c[i] = a[i] * 2 + 1;`,
    caseTitle: "XLA 與 Inductor",
    case: "XLA 與 TorchInductor 都會融合多個張量運算，以避免產生中間張量，並減少記憶體讀寫。",
    related: ["operator-fusion", "loop-tiling", "linalg-dialect", "torch-inductor"],
    sources: [["MLIR: Linalg Dialect", "https://mlir.llvm.org/docs/Dialects/Linalg/"]],
    papers: [TVM_PAPER("討論如何依圖結構融合運算子。")],
  }),
  c({
    id: "strength-reduction",
    title: "Strength Reduction",
    zh: "運算強度折減",
    summary: "把成本高的運算換成成本低的等價運算，例如以位移取代乘以 2 的次方。",
    intuition: "要算 8 倍，不用一步步慢慢乘，直接把小數點往左移幾位，就能得到結果。",
    mechanism: [
      "以代價較低的運算取代代價較高的運算，例如以移位取代乘法。",
      "在迴圈中，可把每次重算的乘法改成累加。",
      "機器學習圖上也有類似改寫，例如把 x ** 2 換成 x * x。",
    ],
    diagram: ["高成本運算", "比對改寫規則", "低成本等價運算"],
    code: `// 優化前
for (int i = 0; i < n; i++) a[i] = b[i * 4];
// 優化後：以累加取代乘法
for (int i = 0, k = 0; i < n; i++, k += 4) a[i] = b[k];`,
    caseTitle: "LLVM 的相關 pass",
    case: "LLVM 的 instcombine 與 loop-reduce 等 pass 都有類似的最佳化。",
    related: ["peephole-optimization", "constant-folding", "llvm", "llvm-pass-manager"],
    sources: [LLVM_PASSES],
    papers: [p("Operator Strength Reduction", "TOPLAS", 2001, "https://doi.org/10.1145/504709.504710", "在 SSA 形式上進行運算強度折減的演算法。")],
  }),
  c({
    id: "llvm-pass-manager",
    title: "Compiler Pass Manager",
    zh: "編譯器 Pass 管理器",
    summary: "負責安排最佳化 pass 的順序與範圍，並管理分析結果的快取與失效。",
    intuition: "工地總管決定工序順序，並記錄哪些檢查結果在施工後已經不能再用。",
    mechanism: [
      "Pass 可在 module、function 或 loop 等不同層級上執行。",
      "分析結果會被快取，直到被改動它的 pass 標記為失效。",
      "pass 的順序會影響結果，因為一個 pass 可能為另一個 pass 創造機會。",
    ],
    diagram: ["Pass pipeline", "層級管理", "分析快取", "失效與重算"],
    code: `# 指定 pipeline 執行，並印出耗時
opt -passes='function(mem2reg,instcombine,simplifycfg)' \\
  -time-passes input.ll -S -o output.ll`,
    caseTitle: "LLVM 與 MLIR",
    case: "LLVM 的 New Pass Manager 與 MLIR 的 PassManager，都提供 pipeline 與分析結果管理。",
    related: ["llvm", "mlir", "dead-code-elimination", "strength-reduction"],
    sources: [["LLVM: Using the New Pass Manager", "https://llvm.org/docs/NewPassManager.html"], ["MLIR: Pass Infrastructure", "https://mlir.llvm.org/docs/PassManagement/"]],
    papers: [LLVM_PAPER("說明 LLVM 以 pass 組成最佳化流程的設計。")],
  }),
  c({
    id: "tensor-comprehensions",
    title: "Tensor Comprehensions",
    zh: "Tensor Comprehensions 語言",
    summary: "以數學形式描述張量運算的語言與編譯器，運用多面體技術，自動產生高效的程式。",
    intuition: "把公式照數學寫下來，至於迴圈怎麼排、怎麼上 GPU，都交給編譯器決定。",
    mechanism: [
      "以 Einstein notation 風格的語法描述運算。",
      "編譯器推導範圍，並以多面體方法進行排程。",
      "再產生 CPU 或 GPU 的程式碼。",
    ],
    diagram: ["張量運算式", "範圍推導", "多面體排程", "產生 GPU 程式"],
    code: `def mm(float(M, K) A, float(K, N) B) -> (float(M, N) C) {
  C(m, n) +=! A(m, k) * B(k, n);
}`,
    caseTitle: "早期的多面體 ML 編譯器",
    case: "Tensor Comprehensions 由 Facebook AI Research 等研究者提出，是把多面體方法用於深度學習的早期工作。",
    related: ["polyhedral-model", "halide", "tiramisu", "taco-compiler"],
    sources: [["Tensor Comprehensions (GitHub)", "https://github.com/facebookresearch/TensorComprehensions"]],
    papers: [p("Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions", "arXiv", 2018, "https://arxiv.org/abs/1802.04730", "Tensor Comprehensions 的原始論文。")],
  }),
  c({
    id: "tiramisu",
    title: "Tiramisu",
    zh: "Tiramisu 多面體編譯器",
    summary: "以多層 IR 把演算法、迴圈排程、資料布局與通訊分開的多面體編譯器。",
    intuition: "蓋房子時把結構、裝潢和水電分開規劃，改裝潢時就不必動結構。",
    mechanism: [
      "演算法層只描述運算，不涉及排程與記憶體。",
      "接著依序指定迴圈排程、資料布局與通訊。",
      "以多面體表示，能處理較複雜的迴圈轉換，並支援多種硬體目標。",
    ],
    diagram: ["演算法層", "迴圈排程層", "資料布局層", "通訊層", "程式碼生成"],
    code: `// Tiramisu C++ API 示意
var i("i", 0, N), i0("i0"), i1("i1");
computation c({i}, expr(i) * 2);
c.tile(i, 32, i0, i1);    // 只改排程，不改演算法
c.parallelize(i0);`,
    caseTitle: "多層 IR 的研究",
    case: "Tiramisu 由 MIT 研究者提出，延續演算法與排程分離的概念，並進一步把資料布局與通訊也分開。",
    related: ["halide", "polyhedral-model", "tensor-comprehensions", "ansor"],
    sources: [["Tiramisu (GitHub)", "https://github.com/Tiramisu-Compiler/tiramisu"]],
    papers: [p("Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code", "CGO", 2019, "https://arxiv.org/abs/1804.10694", "Tiramisu 的原始論文。")],
  }),
];
