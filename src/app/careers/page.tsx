"use client";

import Link from "next/link";
import { careerPaths } from "@/data/careerPaths";
import { concepts } from "@/data/concepts";

export default function CareersPage() {
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
              className="text-slate-700 hover:text-slate-900 font-medium text-brand-600"
            >
              Career Paths
            </Link>
            <Link
              href="/pipeline"
              className="text-slate-700 hover:text-slate-900 font-medium"
            >
              CI/CD
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Title */}
        <div className="mb-12">
          <div className="eyebrow">TECHNICAL CAREER EXPLORER</div>
          <h1 className="text-4xl font-bold mb-4">
            AI Engineering Career Paths
          </h1>
          <p className="text-lg text-slate-600 max-w-3xl">
            Explore seven specialized technical career directions in AI
            Engineering, with context for Singapore and Taiwan markets.
            <strong className="block mt-2 text-slate-900">
              Disclaimer: Evaluations are editorial. Market claims require
              current verification. This is not a salary ranking or job
              guarantee.
            </strong>
          </p>
        </div>

        {/* Career Paths Grid */}
        <div className="space-y-12">
          {careerPaths.map((path) => {
            const pathConcepts = path.conceptIds
              .map((id) => concepts.find((c) => c.id === id))
              .filter((c) => c !== undefined);

            return (
              <section
                key={path.id}
                className={`border-2 rounded-lg p-8 ${
                  path.focus === "recommended"
                    ? "border-brand-300 bg-brand-50"
                    : "border-slate-200"
                }`}
              >
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-slate-900 mb-2">
                      {path.title}
                    </h2>
                    {path.focus === "recommended" && (
                      <span className="inline-block px-3 py-1 bg-brand-600 text-white text-sm font-medium rounded mb-3">
                        Recommended Path
                      </span>
                    )}
                  </div>
                  <span className="text-3xl font-bold text-slate-300">
                    #{path.priority}
                  </span>
                </div>

                <p className="text-slate-700 mb-6 text-lg">{path.summary}</p>

                {/* Markets */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 mb-2">
                      Singapore Market
                    </h3>
                    <p className="text-slate-600 text-sm">{path.markets.singapore}</p>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 mb-2">
                      Taiwan Market
                    </h3>
                    <p className="text-slate-600 text-sm">{path.markets.taiwan}</p>
                  </div>
                </div>

                {/* Job Roles */}
                <div className="mb-8">
                  <h3 className="text-sm font-semibold text-slate-700 mb-3">
                    Typical Job Titles
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {path.roles.map((role) => (
                      <span
                        key={role}
                        className="px-3 py-1 bg-slate-100 text-slate-700 text-sm rounded-full"
                      >
                        {role}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Core Skills */}
                <div className="mb-8">
                  <h3 className="text-sm font-semibold text-slate-700 mb-3">
                    Core Skills
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {path.skills.slice(0, 10).map((skill) => (
                      <span
                        key={skill}
                        className="px-3 py-1 bg-slate-100 text-slate-700 text-sm rounded"
                      >
                        {skill}
                      </span>
                    ))}
                    {path.skills.length > 10 && (
                      <span className="text-sm text-slate-500 px-3 py-1">
                        +{path.skills.length - 10} more
                      </span>
                    )}
                  </div>
                </div>

                {/* Related Concepts */}
                {pathConcepts.length > 0 && (
                  <div className="mb-8">
                    <h3 className="text-sm font-semibold text-slate-700 mb-3">
                      Related Encyclopedia Entries
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {pathConcepts.map((concept) => (
                        <Link
                          key={concept.id}
                          href={`/concepts/${concept.id}`}
                          className="text-sm text-brand-600 hover:text-brand-700 hover:underline p-2 rounded hover:bg-brand-100 transition"
                        >
                          {concept.title}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {/* Assessment */}
                {(path.tradeoffs || path.assessment) && (
                  <div className="pt-6 border-t border-slate-200 space-y-4">
                    {path.tradeoffs && (
                      <div>
                        <h4 className="text-sm font-semibold text-slate-700 mb-1">
                          Trade-offs & Considerations
                        </h4>
                        <p className="text-sm text-slate-600">{path.tradeoffs}</p>
                      </div>
                    )}
                    {path.assessment && (
                      <div>
                        <h4 className="text-sm font-semibold text-slate-700 mb-1">
                          Editorial Assessment
                        </h4>
                        <p className="text-sm text-slate-600">{path.assessment}</p>
                      </div>
                    )}
                  </div>
                )}
              </section>
            );
          })}
        </div>

        {/* Footer Info */}
        <section className="mt-16 p-8 bg-slate-50 border border-slate-200 rounded-lg">
          <h2 className="font-semibold text-slate-900 mb-4">About These Career Paths</h2>
          <ul className="space-y-3 text-sm text-slate-600">
            <li>
              <strong>Editorial guidance:</strong> These paths represent the
              editors&apos; assessment of growing technical specializations. They are
              not exhaustive and may change as the field evolves.
            </li>
            <li>
              <strong>Market information:</strong> Singapore and Taiwan
              opportunities are based on public industry data and job listings.
              Verify current market conditions before making career decisions.
            </li>
            <li>
              <strong>Not a guarantee:</strong> Technical durability, salary
              levels, and job availability are not guaranteed. The field of AI
              is evolving rapidly, and automation may affect these paths.
            </li>
            <li>
              <strong>Explore the atlas:</strong> Click any concept name above
              to dive into technical details, code examples, and industry
              applications.
            </li>
          </ul>
        </section>
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
