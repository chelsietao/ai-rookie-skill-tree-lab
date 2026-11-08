"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { concepts } from "@/data/concepts";
import { searchConcepts, getCategories, filterByCategory } from "@/lib/search";
import ConceptCard from "@/components/ConceptCard";
import ConceptGraph from "@/components/ConceptGraph";

export default function HomePage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [activeView, setActiveView] = useState<"home" | "list">("home");

  const categories = useMemo(() => getCategories(concepts), []);

  const filteredConcepts = useMemo(() => {
    let result = searchConcepts(concepts, searchQuery);
    if (selectedCategory) {
      result = filterByCategory(result, selectedCategory);
    }
    return result;
  }, [searchQuery, selectedCategory]);

  const publishedConcepts = useMemo(
    () =>
      filteredConcepts.filter(
        (c) => c.status === "published" || c.status === "reviewed"
      ),
    [filteredConcepts]
  );

  const focusConcepts = useMemo(
    () => concepts.filter((c) => c.level === "Core focus"),
    []
  );

  const handleReset = () => {
    setSearchQuery("");
    setSelectedCategory("");
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
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
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        {activeView === "home" ? (
          <>
            {/* Hero Section */}
            <section className="mb-16">
              <div className="eyebrow">EXPLORE THE SYSTEMS BEHIND AI</div>
              <h1 className="text-5xl font-bold mb-4 leading-tight">
                Understand how <em className="italic">AI actually works.</em>
              </h1>
              <p className="text-lg text-slate-600 mb-4 max-w-2xl">
                從 Python、機器學習、LLM 到 GPU Kernel、Inference Optimization
                與 Compiler。點擊任何技術名詞，深入理解它的原理及企業應用。
              </p>
              <div className="flex gap-8 text-sm text-slate-600 mb-12">
                <span>
                  <strong className="text-slate-900">
                    {concepts.length}
                  </strong>{" "}
                  concepts
                </span>
                <span>
                  <strong className="text-slate-900">
                    {publishedConcepts.length}
                  </strong>{" "}
                  deep-dive articles
                </span>
                <span>Open reference atlas</span>
              </div>
            </section>

            {/* Search & Filters */}
            <section className="mb-8">
              <div className="flex gap-4 items-center mb-4">
                <div className="flex-1">
                  <input
                    type="text"
                    placeholder="搜尋技術名詞，如 KV Cache、Triton、Attention..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
                  />
                </div>
                {(searchQuery || selectedCategory) && (
                  <button
                    onClick={handleReset}
                    className="px-4 py-2 text-sm text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition"
                  >
                    Reset
                  </button>
                )}
              </div>

              {/* Category Filter */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedCategory("")}
                  className={`px-3 py-1 text-sm rounded-lg border transition ${
                    selectedCategory === ""
                      ? "border-brand-500 bg-brand-50 text-brand-700"
                      : "border-slate-200 text-slate-600 hover:border-slate-300"
                  }`}
                >
                  All
                </button>
                {categories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-3 py-1 text-sm rounded-lg border transition ${
                      selectedCategory === cat
                        ? "border-brand-500 bg-brand-50 text-brand-700"
                        : "border-slate-200 text-slate-600 hover:border-slate-300"
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </section>

            {/* Priority Areas */}
            <section className="mb-16">
              <div className="mb-8">
                <div className="eyebrow">PRIORITY AREAS</div>
                <h2 className="text-2xl font-bold">Focus on infrastructure.</h2>
                <p className="text-slate-600 mt-2">
                  Browse by area or follow relationships between concepts.
                </p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {focusConcepts.slice(0, 8).map((concept) => (
                  <Link
                    key={concept.id}
                    href={`/concepts/${concept.id}`}
                    className="p-4 border border-slate-200 rounded-lg hover:border-brand-400 hover:bg-brand-50 transition"
                  >
                    <div className="font-semibold text-slate-900">
                      {concept.title}
                    </div>
                    <div className="text-sm text-slate-600 mt-1">
                      {concept.zh}
                    </div>
                  </Link>
                ))}
              </div>
            </section>

            {/* Knowledge Graph */}
            <section className="mb-16">
              <div className="mb-8">
                <div className="eyebrow">KNOWLEDGE MAP</div>
                <h2 className="text-2xl font-bold">Explore connections.</h2>
                <p className="text-slate-600 mt-2">
                  Click any node to read its article.
                </p>
              </div>

              <div className="h-96 border border-slate-200 rounded-lg overflow-hidden bg-slate-50">
                <ConceptGraph concepts={filteredConcepts} />
              </div>
            </section>

            {/* CTA */}
            <section className="text-center py-12 border-t border-slate-200">
              <h2 className="text-2xl font-bold mb-4">Ready to explore?</h2>
              <p className="text-slate-600 mb-6 max-w-2xl mx-auto">
                Browse all concepts, discover career pathways, or dive into
                specific technical domains.
              </p>
              <div className="flex justify-center gap-4">
                <button
                  onClick={() => setActiveView("list")}
                  className="px-6 py-2 bg-brand-500 text-white rounded-lg hover:bg-brand-600 transition"
                >
                  View All Concepts
                </button>
                <Link
                  href="/careers"
                  className="px-6 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition"
                >
                  Career Paths
                </Link>
              </div>
            </section>
          </>
        ) : (
          <>
            {/* List View */}
            <section>
              <div className="mb-8">
                <div className="eyebrow">THE KNOWLEDGE LIBRARY</div>
                <h2 className="text-2xl font-bold mb-4">All concepts</h2>
                <p className="text-slate-600">
                  {filteredConcepts.length} concepts found
                </p>
              </div>

              {filteredConcepts.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredConcepts.map((concept) => (
                    <ConceptCard key={concept.id} concept={concept} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <p className="text-slate-600">
                    找不到符合的技術名詞，請改用其他關鍵字。
                  </p>
                </div>
              )}
            </section>

            <div className="mt-8">
              <button
                onClick={() => setActiveView("home")}
                className="text-brand-600 hover:text-brand-700 font-medium"
              >
                ← Back to overview
              </button>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-slate-600">
          <p>
            AI Engineering Atlas · Original educational reference · Official
            sources linked in each entry
          </p>
        </div>
      </footer>
    </div>
  );
}
