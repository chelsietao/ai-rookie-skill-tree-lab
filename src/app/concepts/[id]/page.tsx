import Link from "next/link";
import { redirect } from "next/navigation";
import { concepts } from "@/data/concepts";
import { getRelatedConcepts } from "@/lib/search";

interface PageProps {
  params: Promise<{ id: string }>;
}

export async function generateStaticParams() {
  return concepts.map((concept) => ({
    id: concept.id,
  }));
}

export default async function ConceptPage({ params }: PageProps) {
  const { id } = await params;
  const concept = concepts.find((c) => c.id === id);

  if (!concept) {
    redirect("/");
  }

  const relatedConcepts = getRelatedConcepts(concept, concepts);

  const statusLabel = {
    glossary: "Glossary Entry",
    draft: "Draft",
    reviewed: "Reviewed",
    published: "Published",
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-slate-200 sticky top-0 z-40 bg-white">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
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
              className="text-slate-700 hover:text-slate-900 font-medium"
            >
              CI/CD
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Breadcrumb */}
        <Link href="/" className="text-sm text-slate-600 hover:text-slate-900">
          ← Back to atlas
        </Link>

        {/* Title Section */}
        <div className="mt-8 mb-12">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div className="flex-1">
              <h1 className="text-4xl font-bold text-slate-900 mb-2">
                {concept.title}
              </h1>
              <p className="text-lg text-slate-600">{concept.zh}</p>
            </div>
            <div className="text-right">
              <span className="inline-block px-3 py-1 bg-slate-100 text-slate-700 rounded text-sm font-medium">
                {concept.category}
              </span>
              <div className="text-xs text-slate-500 mt-2">
                {statusLabel[concept.status]}
              </div>
            </div>
          </div>

          {/* Summary */}
          <p className="text-lg text-slate-700 mb-6 pb-6 border-b border-slate-200">
            {concept.summary}
          </p>
        </div>

        {/* Content Sections */}
        <article className="prose-markdown mb-16">
          {/* Section 1: What Is It? */}
          {concept.intuition && (
            <section className="mb-12">
              <h2>What Is It?</h2>
              <p>{concept.intuition}</p>
            </section>
          )}

          {/* Section 2: How Does It Actually Work? */}
          {concept.mechanism && concept.mechanism.length > 0 && (
            <section className="mb-12">
              <h2>How Does It Actually Work?</h2>
              <ul>
                {concept.mechanism.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </section>
          )}

          {/* Section 3: Diagram */}
          {concept.diagram && concept.diagram.length > 0 && (
            <section className="mb-12">
              <h2>Architecture Diagram</h2>
              <div className="p-6 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="space-y-2 font-mono text-sm">
                  {concept.diagram.map((item, i) => (
                    <div key={i} className="text-slate-700">
                      {i > 0 && (
                        <span className="text-slate-400 mr-2">↓</span>
                      )}
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* Section 4: Code Example */}
          {concept.code && (
            <section className="mb-12">
              <h2>Code Example</h2>
              <pre>
                <code>{concept.code}</code>
              </pre>
            </section>
          )}

          {/* Section 5: Industry Case */}
          {concept.case && (
            <section className="mb-12">
              <h2>{concept.caseTitle || "Industry Case Study"}</h2>
              <p>{concept.case}</p>
            </section>
          )}

          {/* Section 6: Related Concepts */}
          {relatedConcepts.length > 0 && (
            <section className="mb-12">
              <h2>Related Concepts</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {relatedConcepts.map((c) => (
                  <Link
                    key={c.id}
                    href={`/concepts/${c.id}`}
                    className="p-3 border border-slate-200 rounded-lg hover:border-brand-400 hover:bg-brand-50 transition"
                  >
                    <div className="font-semibold text-slate-900 text-sm">
                      {c.title}
                    </div>
                    <div className="text-xs text-slate-600">{c.zh}</div>
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* Section 7: Official Resources */}
          {concept.sources && concept.sources.length > 0 && (
            <section className="mb-12">
              <h2>Official Resources</h2>
              <ul>
                {concept.sources.map((source, i) => (
                  <li key={i}>
                    <a href={source[1]} target="_blank" rel="noopener noreferrer">
                      {source[0]}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </article>

        {/* Footer Navigation */}
        <div className="border-t border-slate-200 pt-8">
          <Link href="/" className="text-brand-600 hover:text-brand-700 font-medium">
            ← Back to atlas
          </Link>
        </div>
      </main>
    </div>
  );
}
