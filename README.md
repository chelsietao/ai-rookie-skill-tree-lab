# AI Engineering Atlas：AI 工程技術名詞知識圖譜

線上網站：https://chelsietao.github.io/ai-rookie-skill-tree-lab/

AI Engineering Atlas 是一個以繁體中文撰寫的 AI 工程技術名詞百科。專題整理了從模型訓練、推論優化、編譯器到 GPU 運算與分散式系統的 100 個技術概念，並呈現概念之間的關係，讓讀者能從任何一個名詞出發，理解它的原理、用途與相關技術。

## 專題內容

### 100 個技術概念

概念分為 10 個領域：

- Inference Optimization（推論優化）
- Compiler Engineering（編譯器工程）
- GPU Computing（GPU 運算）
- Deep Learning（深度學習）
- Distributed Systems（分散式系統）
- LLM Training（大型語言模型訓練）
- LLM & Agents（大型語言模型與代理）
- Data & MLOps（資料與機器學習維運）
- Machine Learning（機器學習）
- Programming（程式設計）

每個概念依固定結構撰寫，包含：摘要、直覺理解、運作機制、架構圖、程式碼範例、產業案例、相關概念與參考來源。

### 概念關係圖

概念之間以七種關係連結：前置知識、相關、應用於、實作、優化、替代方案、組成部分。首頁的關係圖可用來瀏覽概念之間的脈絡。

### 搜尋與篩選

支援以英文或中文名詞搜尋，並可依領域篩選。

### 七條職業方向

整理七個 AI 工程職業方向，每一條說明所需技能、對應概念、常見職稱，以及新加坡與台灣兩地的市場背景：

1. 推論優化與 AI 編譯器
2. GPU / CUDA 效能工程
3. 分散式訓練與 AI 系統
4. AI 平台與 MLOps
5. 應用機器學習與電腦視覺
6. 大型語言模型應用與 Agent 工程
7. AI 基礎設施維運與網路

職業方向的評估屬於編輯判斷，不是薪資排名，也不構成就業保證。

### CI/CD 說明頁

網站內附 CI/CD 頁面（/pipeline），說明本專案如何以 GitHub Actions 自動建置與部署，並列出四個 workflow 的觸發時機與用途。

### 內容狀態

每個概念標示內容狀態，分為 glossary、draft、reviewed、published 四種，用以區分內容的完整程度。

### 語言原則

介面與說明使用繁體中文，技術術語保留英文，避免翻譯造成誤解。

## 技術架構

- Next.js 15（App Router，靜態匯出）
- React 19
- TypeScript
- Tailwind CSS
- 概念與職業資料以 TypeScript 檔案維護，無需資料庫
- 以 GitHub Actions 部署至 GitHub Pages

## 專案結構

```
src/
  app/
    page.tsx              首頁：概念關係圖、搜尋、領域篩選
    concepts/             概念列表與單一概念頁面
    careers/              職業方向頁面
  components/             ConceptCard、ConceptGraph
  data/
    concepts.ts           彙整所有概念
    concepts/             依領域拆分的概念資料
    careerPaths.ts        七條職業方向
  lib/search.ts           搜尋與篩選邏輯
  types/index.ts          型別定義
```

## 在本機執行

需要 Node.js 18 以上。

```bash
npm install
npm run dev
```

開啟 http://localhost:3000/ai-rookie-skill-tree-lab/ 即可瀏覽。

建置靜態網站：

```bash
npm run build
```

輸出位於 `out/` 目錄。

## 授權

MIT
