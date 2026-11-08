"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { concepts } from "@/data/concepts";
import { searchConcepts, getCategories, filterByCategory } from "@/lib/search";
import ConceptCard from "@/components/ConceptCard";

export default function ConceptsPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");

  const categories = useMemo(() => getCategories(concepts), []);

  const filteredConcepts = useMemo(() => {
    let result = searchConcepts(concepts, searchQuery);
    if (selectedCategory) {
      result = filterByCategory(result, selectedCategory);
    }
    return result;
  }, [searchQuery, selectedCategory]);

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
              className="text-slate-700 hover:text-slate-900 font-medium text-brand-600"
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

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Title */}
        <div className="mb-12">
          <div className="eyebrow">THE KNOWLEDGE LIBRARY</div>
          <h1 className="text-4xl font-bold mb-4">All Concepts</h1>
          <p className="text-lg text-slate-600">
            Explore {concepts.length} concepts across AI Engineering disciplines.
          </p>
        </div>

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

        {/* Results */}
        {filteredConcepts.length > 0 ? (
          <>
            <p className="text-slate-600 mb-6">
              Found {filteredConcepts.length} concept
              {filteredConcepts.length !== 1 ? "s" : ""}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
              {filteredConcepts.map((concept) => (
                <ConceptCard key={concept.id} concept={concept} />
              ))}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <p className="text-slate-600 mb-4">
              找不到符合的技術名詞，請改用其他關鍵字。
            </p>
            <button
              onClick={handleReset}
              className="text-brand-600 hover:text-brand-700 font-medium"
            >
              Clear filters
            </button>
          </div>
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
