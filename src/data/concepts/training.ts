import type { Concept } from "@/types";

export const trainingConcepts: Concept[] = [
  {
    id: "pretraining",
    title: "Pretraining",
    zh: "預訓練",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "在大規模未標註文字上以下一個 token 預測等自監督目標訓練語言模型，使其學得語言與世界知識的通用表示。",
    intuition:
      "像讓學生先廣泛閱讀大量書籍，培養語感與常識，之後再針對特定任務做專門訓練。",
    mechanism: [
      "收集、清理並去重大規模文字語料，經過分詞器轉成 token 序列。",
      "以 causal language modeling 為目標：給定前面的 token，預測下一個 token，損失為交叉熵。",
      "透過資料平行、張量平行、管線平行等組合在大量 GPU 上訓練，並使用學習率排程、混合精度與梯度裁剪維持穩定。",
      "訓練過程需要持續監控損失、評估集表現與異常（例如損失尖峰），並定期保存檢查點以便恢復。",
    ],
    diagram: [
      "語料收集與清理",
      "分詞並打包序列",
      "分散式訓練（下一 token 預測）",
      "監控與保存檢查點",
      "得到基礎模型（base model）",
    ],
    code: `import torch.nn.functional as F

# logits: [B, T, V], input_ids: [B, T]
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    input_ids[:, 1:].reshape(-1),
)`,
    caseTitle: "GPT-3 與 Llama 2 的公開技術報告",
    case: "GPT-3 論文展示了以大規模預訓練得到的少樣本能力；Llama 2 的技術報告則公開描述了其預訓練與後續對齊的整體流程。各組織的資料與細節並非全部公開，不應假設它們使用相同做法。",
    related: ["scaling-laws", "transformer", "tokenization", "sft"],
    sources: [
      ["Language Models are Few-Shot Learners", "https://arxiv.org/abs/2005.14165"],
      ["Llama 2 paper", "https://arxiv.org/abs/2307.09288"],
    ],
  },
  {
    id: "sft",
    title: "Supervised Fine-Tuning",
    zh: "監督式微調（指令微調）",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "以人工撰寫或篩選的「指令—回應」範例繼續訓練預訓練模型，使其學會遵循指令與特定的回答風格。",
    intuition:
      "像讓博學的新人看大量「問題與標準答案」的範例，學會怎樣把知識整理成別人想要的回答。",
    mechanism: [
      "準備指令與期望回應的資料，並套用統一的對話模板（chat template）轉成 token 序列。",
      "以與預訓練相同的下一 token 預測損失訓練，但通常只對回應部分計算損失（遮蔽提示部分）。",
      "學習率通常比預訓練小，訓練步數少，資料的品質與多樣性往往比數量更關鍵。",
      "SFT 之後常再接偏好最佳化（如 RLHF 或 DPO），以進一步調整回答的偏好與安全性。",
    ],
    diagram: [
      "指令—回應資料",
      "套用對話模板",
      "只對回應 token 計算損失",
      "微調訓練",
      "得到指令模型",
    ],
    code: `# 只對回應部分計算損失：把提示部分的 label 設為 -100
labels = input_ids.clone()
labels[:, :prompt_len] = -100
loss = model(input_ids=input_ids, labels=labels).loss`,
    caseTitle: "FLAN 與 InstructGPT",
    case: "FLAN 論文研究了以多任務指令資料微調提升零樣本表現；InstructGPT 論文公開了以人類示範資料做監督微調，再接獎勵模型與強化學習的流程。",
    related: ["pretraining", "lora", "rlhf", "dpo"],
    sources: [
      ["FLAN: Finetuned Language Models Are Zero-Shot Learners", "https://arxiv.org/abs/2109.01652"],
      ["InstructGPT paper", "https://arxiv.org/abs/2203.02155"],
    ],
  },
  {
    id: "lora",
    title: "LoRA",
    zh: "低秩適配",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "凍結原模型權重，只訓練插入在權重旁的低秩矩陣，大幅減少需要訓練與保存的參數量。",
    intuition:
      "像不重寫整本書，只在旁邊貼幾張便利貼修正重點；便利貼很薄，要換成另一組也很容易。",
    mechanism: [
      "對某個權重矩陣 W（d×k），不更新 W，而是學習更新量 ΔW = B·A，其中 A 為 r×k、B 為 d×r，r 遠小於 d 與 k。",
      "前向時輸出為 Wx + (α/r)·BAx；B 通常初始化為零，因此訓練開始時模型行為與原模型一致。",
      "可訓練參數量與優化器狀態大幅下降，且不同任務可保存各自的小型 adapter，部署時可合併回 W 以避免額外推論延遲。",
      "秩 r、套用的層（常為注意力的投影矩陣）與 α 是主要超參數，效果依任務與資料而異。",
    ],
    diagram: [
      "凍結的權重 W",
      "並行的低秩 A、B 矩陣",
      "輸出 = Wx + BAx 的縮放和",
      "只對 A、B 反向傳播",
      "儲存 adapter（可合併回 W）",
    ],
    code: `from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, cfg)
model.print_trainable_parameters()`,
    caseTitle: "LoRA 論文與 PEFT",
    case: "LoRA 論文報告了在多個模型與任務上以少量可訓練參數達到接近全參數微調的結果；Hugging Face PEFT 函式庫提供了實作與文件。適合與否取決於任務難度與資料量。",
    related: ["qlora", "sft", "quantization", "gradient-checkpointing"],
    sources: [
      ["LoRA paper", "https://arxiv.org/abs/2106.09685"],
      ["Hugging Face PEFT documentation", "https://huggingface.co/docs/peft/index"],
    ],
  },
  {
    id: "qlora",
    title: "QLoRA",
    zh: "量化低秩適配",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "把凍結的基礎模型量化為 4 位元，再在其上訓練 LoRA adapter，讓大型模型能在較少的 GPU 記憶體下微調。",
    intuition:
      "像把一本厚書縮印成小開本放在手邊，只在旁邊用較精確的筆記寫下修改；書很省空間，筆記仍然準確。",
    mechanism: [
      "基礎模型權重以 4 位元的 NormalFloat（NF4）格式量化並凍結，使用雙重量化降低量化常數的額外記憶體。",
      "前向與反向傳播時，把需要的權重區塊反量化為較高精度（如 BF16）參與計算，梯度只流向 LoRA adapter。",
      "採用分頁優化器（paged optimizers）處理記憶體尖峰，避免訓練中途記憶體不足。",
      "節省的是記憶體而非計算量，反量化會帶來額外開銷；效果與穩定性需依任務驗證。",
    ],
    diagram: [
      "基礎模型量化為 4-bit NF4",
      "凍結量化權重",
      "反量化後前向計算",
      "梯度只更新 LoRA adapter",
      "輸出 adapter 權重",
    ],
    code: `from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained("your-model", quantization_config=bnb)`,
    caseTitle: "QLoRA 論文",
    case: "QLoRA 論文報告了以 4 位元量化搭配 LoRA 在單張大記憶體 GPU 上微調大型模型的結果，並公開了方法細節。實際可行的模型大小取決於 GPU 記憶體與序列長度。",
    related: ["lora", "quantization", "gpu-memory", "sft"],
    sources: [
      ["QLoRA paper", "https://arxiv.org/abs/2305.14314"],
      ["Hugging Face bitsandbytes integration", "https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes"],
    ],
  },
  {
    id: "rlhf",
    title: "RLHF",
    zh: "人類回饋強化學習",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "先以人類偏好比較資料訓練獎勵模型，再用強化學習（常為 PPO）最佳化語言模型，使輸出更符合人類偏好。",
    intuition:
      "像讓學徒反覆交作品，評審不給標準答案，只說「A 比 B 好」；學徒從評審偏好中學會怎樣的作品更受青睞。",
    mechanism: [
      "先做監督微調得到一個初始策略模型。",
      "收集人類對同一提示的多個回答的偏好排序，訓練獎勵模型預測人類偏好分數。",
      "用 PPO 等強化學習演算法最佳化策略模型以提高獎勵，同時加入與參考模型之間的 KL 懲罰，避免偏離過遠或鑽獎勵漏洞。",
      "流程需要同時載入策略、參考、獎勵（與價值）模型，系統複雜且不穩定，這也是 DPO 等簡化方法出現的動機之一。",
    ],
    diagram: [
      "SFT 模型",
      "收集人類偏好比較資料",
      "訓練獎勵模型",
      "PPO 最佳化（含 KL 懲罰）",
      "對齊後的模型",
    ],
    code: `# 概念性：PPO 階段每個回應的目標獎勵
reward = reward_model(prompt, response)
kl = (logp_policy - logp_ref).sum()
objective = reward - beta * kl`,
    caseTitle: "InstructGPT 與摘要偏好研究",
    case: "InstructGPT 論文公開了 SFT、獎勵模型與 PPO 三階段流程，並報告了人類評估結果；OpenAI 早先的摘要研究也使用了人類回饋。Hugging Face 的部落格文章以淺顯方式說明整體流程。",
    related: ["reward-model", "ppo", "dpo", "sft"],
    sources: [
      ["InstructGPT paper", "https://arxiv.org/abs/2203.02155"],
      ["Learning to summarize from human feedback", "https://arxiv.org/abs/2009.01325"],
      ["Hugging Face: Illustrating RLHF", "https://huggingface.co/blog/rlhf"],
    ],
  },
  {
    id: "ppo",
    title: "Proximal Policy Optimization",
    zh: "近端策略最佳化",
    category: "LLM Training",
    level: "Reference",
    status: "published",
    summary:
      "強化學習演算法，以截斷（clipping）限制每次策略更新的幅度，讓訓練穩定，常用於 RLHF 的策略最佳化階段。",
    intuition:
      "像教練要求選手每次只微調動作一點點，不要一次改太多；步子小而穩，才不會一改就把原本會的都忘了。",
    mechanism: [
      "計算新舊策略的機率比值 r = π_new(a|s) / π_old(a|s)。",
      "目標函數取 min(r·A, clip(r, 1−ε, 1+ε)·A)，其中 A 是優勢（advantage）；比值超出範圍時，梯度被截斷，限制更新幅度。",
      "同一批資料可做多個 epoch 更新，比單純的策略梯度更有樣本效率。",
      "在 RLHF 中，優勢通常由獎勵、KL 懲罰與價值模型估計得到，因此需要額外維護價值網路。",
    ],
    diagram: [
      "以舊策略取樣",
      "計算優勢",
      "計算機率比值並截斷",
      "多輪小步更新",
      "更新舊策略",
    ],
    code: `import torch

def ppo_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = (logp_new - logp_old).exp()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()`,
    caseTitle: "PPO 論文與 RLHF 應用",
    case: "Schulman 等人的 PPO 論文提出截斷目標；InstructGPT 論文使用 PPO 作為 RLHF 的最佳化演算法。後續也出現不需要價值模型或不用強化學習的對齊方法。",
    related: ["rlhf", "reward-model", "dpo", "backpropagation"],
    sources: [
      ["Proximal Policy Optimization Algorithms", "https://arxiv.org/abs/1707.06347"],
      ["InstructGPT paper", "https://arxiv.org/abs/2203.02155"],
    ],
  },
  {
    id: "dpo",
    title: "Direct Preference Optimization",
    zh: "直接偏好最佳化",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "直接用偏好成對資料（較佳回答與較差回答）以分類式損失微調模型，不需要另外訓練獎勵模型或跑強化學習。",
    intuition:
      "像直接告訴學徒「這個比那個好」，讓他調整到更偏向好的、遠離差的，不必先請一位評審寫出一套評分標準。",
    mechanism: [
      "資料為 (prompt, chosen, rejected) 三元組，並需要一個凍結的參考模型（通常為 SFT 模型）。",
      "損失函數依據策略與參考模型對 chosen 與 rejected 的對數機率差，鼓勵策略相對於參考模型更偏向 chosen。",
      "超參數 β 控制偏離參考模型的程度，作用類似 RLHF 中的 KL 懲罰。",
      "實作比 PPO 簡單、穩定；但仍依賴偏好資料品質，且是離線方法，效果與線上強化學習的比較仍有研究討論。",
    ],
    diagram: [
      "偏好資料 (prompt, chosen, rejected)",
      "計算策略與參考模型的對數機率",
      "DPO 損失",
      "反向傳播更新策略",
      "對齊後的模型",
    ],
    code: `import torch.nn.functional as F

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # pi_*/ref_*：策略與參考模型對 chosen(w)、rejected(l) 的序列對數機率
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(margin).mean()`,
    caseTitle: "DPO 論文",
    case: "DPO 論文推導出不需獎勵模型即可最佳化偏好目標的方法，並報告了與以 PPO 為基礎的 RLHF 可相比的結果。Hugging Face TRL 等函式庫提供了實作。",
    related: ["rlhf", "sft", "reward-model", "ppo"],
    sources: [
      ["Direct Preference Optimization", "https://arxiv.org/abs/2305.18290"],
      ["Hugging Face TRL documentation", "https://huggingface.co/docs/trl/index"],
    ],
  },
  {
    id: "reward-model",
    title: "Reward Model",
    zh: "獎勵模型",
    category: "LLM Training",
    level: "Reference",
    status: "published",
    summary:
      "以人類偏好比較資料訓練的模型，輸入提示與回答，輸出代表「人類有多喜歡」的分數，作為強化學習的獎勵訊號。",
    intuition:
      "像訓練一位助理評審：看過很多「A 勝過 B」的判決後，學會替新的作品打分數。",
    mechanism: [
      "資料為同一提示下的多個回答及人類的偏好排序。",
      "常以 Bradley–Terry 模型建立損失：讓較被偏好回答的分數高於較不被偏好者，損失為 −log σ(r_chosen − r_rejected)。",
      "模型架構通常是語言模型加上輸出單一純量的頭，並由 SFT 模型初始化。",
      "獎勵模型只是人類偏好的近似，過度最佳化會出現獎勵駭客（reward hacking），所以需要 KL 約束、人類抽查與持續更新。",
    ],
    diagram: [
      "提示與多個回答",
      "人類偏好標註",
      "訓練獎勵模型（成對損失）",
      "對新回答打分",
      "作為 RL 的獎勵訊號",
    ],
    code: `import torch.nn.functional as F

def reward_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()`,
    caseTitle: "從人類偏好學習獎勵",
    case: "Christiano 等人的論文研究了從人類比較中學習獎勵函數，InstructGPT 將此做法用於語言模型。獎勵模型的偏差與過度最佳化風險在文獻中被廣泛討論。",
    related: ["rlhf", "ppo", "dpo", "llm-as-judge"],
    sources: [
      ["Deep RL from Human Preferences", "https://arxiv.org/abs/1706.03741"],
      ["InstructGPT paper", "https://arxiv.org/abs/2203.02155"],
    ],
  },
  {
    id: "knowledge-distillation",
    title: "Knowledge Distillation",
    zh: "知識蒸餾",
    category: "LLM Training",
    level: "Reference",
    status: "published",
    summary:
      "讓較小的學生模型學習較大教師模型的輸出分佈，以較少的參數保留教師的大部分能力。",
    intuition:
      "像學生不只看標準答案，還參考老師「對各選項有多確定」的想法，因此能學到更細緻的判斷。",
    mechanism: [
      "教師模型對輸入產生輸出機率分佈（soft labels），並可以用溫度 T 使分佈更平滑，揭露類別之間的相似關係。",
      "學生模型的損失通常結合兩部分：與教師分佈的 KL 散度，以及與真實標籤的交叉熵。",
      "在語言模型中，教師也可以是產生訓練資料的來源（以教師輸出作為合成資料訓練學生），與直接匹配機率分佈是不同做法。",
      "學生能力受限於容量與資料，且使用某些模型輸出訓練可能受該模型的使用條款限制。",
    ],
    diagram: [
      "教師模型輸出 soft labels",
      "學生模型前向",
      "計算蒸餾損失（KL + 交叉熵）",
      "更新學生",
      "得到較小的模型",
    ],
    code: `import torch.nn.functional as F

def distill_loss(s_logits, t_logits, labels, T=2.0, alpha=0.5):
    soft = F.kl_div(F.log_softmax(s_logits / T, -1),
                    F.softmax(t_logits / T, -1), reduction="batchmean") * T * T
    hard = F.cross_entropy(s_logits, labels)
    return alpha * soft + (1 - alpha) * hard`,
    caseTitle: "Hinton 蒸餾論文與 DistilBERT",
    case: "Hinton 等人的論文提出以溫度化的 soft labels 做蒸餾；DistilBERT 論文報告了蒸餾後的較小模型在保留多數表現的同時縮小並加速。",
    related: ["quantization", "sft", "pretraining", "latency-throughput"],
    sources: [
      ["Distilling the Knowledge in a Neural Network", "https://arxiv.org/abs/1503.02531"],
      ["DistilBERT", "https://arxiv.org/abs/1910.01108"],
    ],
  },
  {
    id: "gradient-checkpointing",
    title: "Gradient Checkpointing",
    zh: "梯度檢查點（啟動值重算）",
    category: "LLM Training",
    level: "Core focus",
    status: "published",
    summary:
      "訓練時不保存所有中間啟動值，只保存部分檢查點，反向傳播時再重算缺少的部分，以額外計算換取較少的記憶體。",
    intuition:
      "像寫長篇推導時只留下幾個關鍵步驟的草稿，需要回頭檢查時，再從最近的關鍵步驟重新算一遍。",
    mechanism: [
      "反向傳播需要前向的啟動值；一般做法保存全部，記憶體隨層數線性增加。",
      "Checkpointing 只保存某些層的輸入，前向時其餘中間值在用完後丟棄。",
      "反向時對每個區段，先從其檢查點重跑前向得到所需啟動值，再計算梯度。",
      "典型代價是多一次前向計算（訓練時間會增加，幅度依設定而定），換取明顯降低的啟動值記憶體，使更大的 batch、序列長度或模型可以訓練。",
    ],
    diagram: [
      "前向：只保存檢查點",
      "丟棄其他中間啟動值",
      "反向：從檢查點重算區段",
      "計算區段梯度",
      "繼續往前一個區段",
    ],
    code: `import torch
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.relu(x @ w1) @ w2

y = checkpoint(block, x, use_reentrant=False)`,
    caseTitle: "Sublinear Memory Cost 論文",
    case: "Chen 等人的論文提出以重算換取亞線性的記憶體成本；PyTorch 提供 torch.utils.checkpoint，Hugging Face 等框架也提供開關。實際記憶體節省與時間開銷需以自己的模型量測。",
    related: ["backpropagation", "gpu-memory", "lora", "fsdp"],
    sources: [
      ["Training Deep Nets with Sublinear Memory Cost", "https://arxiv.org/abs/1604.06174"],
      ["PyTorch torch.utils.checkpoint", "https://docs.pytorch.org/docs/stable/checkpoint.html"],
    ],
  },
  {
    id: "gradient-accumulation",
    title: "Gradient Accumulation",
    zh: "梯度累積",
    category: "LLM Training",
    level: "Reference",
    status: "published",
    summary:
      "連續執行多個小批次的前向與反向，把梯度累加後才更新一次參數，在不增加單次記憶體的情況下模擬較大的有效批次。",
    intuition:
      "像一次搬不動整批貨，就分幾趟把貨疊在同一個倉位，全部搬完才一起結算。",
    mechanism: [
      "每個 micro-batch 執行前向與反向，梯度會累加到 .grad 而不清零。",
      "累積 N 個 micro-batch 後才呼叫 optimizer.step() 並清零梯度。",
      "有效批次大小 = micro-batch 大小 × 累積步數 × 資料平行的 GPU 數。",
      "損失通常要除以累積步數，才能與用大批次一次計算時的梯度尺度一致；在分散式訓練中要注意避免每個 micro-batch 都做不必要的梯度同步。",
    ],
    diagram: [
      "micro-batch 1 前向與反向",
      "micro-batch 2 前向與反向（梯度累加）",
      "…累積 N 次",
      "optimizer.step()",
      "清零梯度",
    ],
    code: `accum = 4
for i, batch in enumerate(loader):
    loss = model(batch).loss / accum
    loss.backward()
    if (i + 1) % accum == 0:
        optimizer.step()
        optimizer.zero_grad()`,
    caseTitle: "大批次訓練",
    case: "Hugging Face Accelerate 文件說明了梯度累積的實作與分散式下的注意事項；Goyal 等人的大批次訓練研究討論了學習率與批次大小的關係。",
    related: ["learning-rate-schedule", "ddp", "gradient-checkpointing", "mixed-precision"],
    sources: [
      ["Hugging Face Accelerate: gradient accumulation", "https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation"],
      ["Accurate, Large Minibatch SGD", "https://arxiv.org/abs/1706.02677"],
    ],
  },
  {
    id: "learning-rate-schedule",
    title: "Learning Rate Schedule",
    zh: "學習率排程",
    category: "LLM Training",
    level: "Reference",
    status: "published",
    summary:
      "在訓練過程中動態調整學習率，常見做法是先 warmup 再以餘弦或線性方式衰減，以兼顧穩定與收斂。",
    intuition:
      "像開車起步先慢慢加速、巡航時保持速度、接近目的地時逐步減速，才能平穩而準確地到站。",
    mechanism: [
      "Warmup：訓練初期從很小的學習率線性升到目標值，避免初期梯度與優化器統計尚不穩定時更新過大。",
      "衰減：warmup 後以餘弦退火、線性或階梯方式降低學習率，讓後期的更新更細緻。",
      "大批次訓練常見的做法是依批次大小調整學習率並搭配 warmup。",
      "排程的總步數、峰值學習率與最終值都是超參數，需要依模型規模與資料調整；調錯時常見的現象是損失發散或收斂停滯。",
    ],
    diagram: [
      "初始很小的學習率",
      "線性 warmup 到峰值",
      "餘弦或線性衰減",
      "接近最小值",
      "訓練結束",
    ],
    code: `import math

def lr_at(step, total, warmup, peak, min_lr=0.0):
    if step < warmup:
        return peak * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))`,
    caseTitle: "SGDR 與大批次訓練",
    case: "SGDR 論文提出餘弦退火（含重啟）的學習率排程；Goyal 等人的研究說明了大批次訓練中 warmup 的作用。PyTorch 文件列出了內建的排程器。",
    related: ["gradient-descent", "gradient-accumulation", "pretraining", "mixed-precision"],
    sources: [
      ["SGDR: Stochastic Gradient Descent with Warm Restarts", "https://arxiv.org/abs/1608.03983"],
      ["PyTorch: how to adjust learning rate", "https://docs.pytorch.org/docs/stable/optim.html"],
    ],
  },
];
