import Link from "next/link";

const deploySteps = [
  {
    name: "Checkout",
    detail: "取得 main 分支的最新原始碼。",
  },
  {
    name: "Setup Node.js",
    detail: "安裝 Node.js 18，並快取 npm 套件以縮短建置時間。",
  },
  {
    name: "npm ci",
    detail: "依 package-lock.json 安裝完全相同版本的相依套件。",
  },
  {
    name: "npm run build",
    detail:
      "Next.js 以靜態匯出（output: export）產生純 HTML、CSS、JavaScript，輸出到 out 目錄。",
  },
  {
    name: "Upload Pages artifact",
    detail: "將 out 目錄打包為 GitHub Pages 部署用的 artifact。",
  },
  {
    name: "Deploy to GitHub Pages",
    detail: "另一個 job 取得 artifact 並發布，網站隨即更新。",
  },
];

const workflows = [
  {
    file: "deploy.yml",
    name: "Deploy to GitHub Pages",
    trigger: "推送到 main，或手動執行",
    purpose: "建置本網站並發布到 GitHub Pages。",
  },
  {
    file: "lint-test.yml",
    name: "lint-test",
    trigger: "每次 push 與 pull request",
    purpose:
      "針對專案的 Python 程式執行 ruff 檢查、驗證 notebook 是否與產生腳本一致，並以 pytest 執行測試與涵蓋率統計。",
  },
  {
    file: "mysql-smoke.yml",
    name: "mysql-smoke",
    trigger: "每次 push 與 pull request",
    purpose:
      "啟動 MySQL 8.4 服務容器，載入專案的資料表結構（schema.sql），並確認 customers 資料表已成功建立。",
  },
  {
    file: "full-lab-smoke.yml",
    name: "full-lab-smoke",
    trigger: "手動執行，或每月 1 日自動執行",
    purpose: "以 portfolio 設定完整重現專案的實驗流程，確認結果可重現。",
  },
];

export default function PipelinePage() {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-slate-200 sticky top-0 z-40 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-2xl">✳️</span>
            <div className="flex flex-col">
              <div className="text-sm font-bold text-slate-900">
                AI Engineering
              </div>
              <div className="text-xs text-slate-500 -mt-1">Atlas</div>
            </div>
          </Link>
          <nav className="flex gap-6 text-sm">
            <Link
              href="/"
              className="text-slate-700 hover:text-slate-900 font-medium"
            >
              Overview
            </Link>
            <Link
              href="/concepts"
              className="text-slate-700 hover:text-slate-900 font-medium"
            >
              All Concepts
            </Link>
            <Link
              href="/careers"
              className="text-slate-700 hover:text-slate-900 font-medium"
            >
              Career Paths
            </Link>
            <Link
              href="/pipeline"
              className="text-slate-700 hover:text-slate-900 font-medium text-brand-600"
            >
              CI/CD
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-12">
          <div className="eyebrow">CI / CD</div>
          <h1 className="text-4xl font-bold mb-4">
            本網站如何自動建置與部署
          </h1>
          <p className="text-lg text-slate-600">
            CI（持續整合）是每次程式變更後自動執行檢查與測試；CD（持續部署）是檢查通過後自動把成果發布出去。本專案使用 GitHub
            Actions 完成這兩件事：只要把變更推送到 main 分支，網站就會自動重新建置並更新，不需要手動上傳檔案。
          </p>
        </div>

        <section className="mb-14">
          <h2 className="text-2xl font-bold mb-2">部署流程</h2>
          <p className="text-slate-600 mb-6">
            網站由 Next.js 靜態匯出，因此部署結果只是一組靜態檔案，由 GitHub Pages 直接提供。流程分為 build 與 deploy
            兩個 job。
          </p>
          <ol className="space-y-3">
            {deploySteps.map((step, i) => (
              <li
                key={step.name}
                className="flex gap-4 border border-slate-200 rounded-lg p-4"
              >
                <span className="text-2xl font-bold text-slate-300 w-8 shrink-0">
                  {i + 1}
                </span>
                <div>
                  <div className="font-semibold text-slate-900">
                    {step.name}
                  </div>
                  <div className="text-sm text-slate-600">{step.detail}</div>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section className="mb-14">
          <h2 className="text-2xl font-bold mb-2">四個 Workflow</h2>
          <p className="text-slate-600 mb-6">
            儲存庫共有四個 workflow，各自負責不同的檢查或發布工作，設定檔位於 .github/workflows。
          </p>
          <div className="space-y-4">
            {workflows.map((w) => (
              <div
                key={w.file}
                className="border border-slate-200 rounded-lg p-5"
              >
                <div className="flex flex-wrap items-baseline gap-3 mb-2">
                  <h3 className="font-semibold text-slate-900">{w.name}</h3>
                  <code className="text-xs text-slate-500">{w.file}</code>
                </div>
                <div className="text-sm text-slate-700 mb-1">
                  <span className="font-medium">觸發時機：</span>
                  {w.trigger}
                </div>
                <div className="text-sm text-slate-600">{w.purpose}</div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2 className="text-2xl font-bold mb-2">原始碼與執行紀錄</h2>
          <ul className="list-disc pl-6 text-slate-700 space-y-1">
            <li>
              <a
                className="text-brand-600 hover:underline"
                href="https://github.com/chelsietao/ai-rookie-skill-tree-lab/actions"
              >
                GitHub Actions 執行紀錄
              </a>
            </li>
            <li>
              <a
                className="text-brand-600 hover:underline"
                href="https://github.com/chelsietao/ai-rookie-skill-tree-lab/tree/main/.github/workflows"
              >
                Workflow 設定檔
              </a>
            </li>
          </ul>
        </section>
      </main>
    </div>
  );
}
