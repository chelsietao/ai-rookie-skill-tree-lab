import Link from "next/link";
import type { Concept } from "@/types";

interface ConceptCardProps {
  concept: Concept;
}

export default function ConceptCard({ concept }: ConceptCardProps) {
  const statusColor = {
    glossary: "bg-slate-100 text-slate-600",
    draft: "bg-yellow-100 text-yellow-600",
    reviewed: "bg-blue-100 text-blue-600",
    published: "bg-green-100 text-green-600",
  };

  return (
    <Link href={`/concepts/${concept.id}`} className="block group">
      <div className="p-4 border border-slate-200 rounded-lg hover:border-brand-500 hover:bg-brand-50 hover:shadow-md transition cursor-pointer h-full bg-white">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex-1">
            <h3 className="font-semibold text-slate-900 group-hover:text-brand-600 transition">
              {concept.title}
            </h3>
            <p className="text-sm text-slate-600 group-hover:text-slate-700">{concept.zh}</p>
          </div>
          <span className={`text-xs px-2 py-1 rounded ${statusColor[concept.status]}`}>
            {concept.level}
          </span>
        </div>

        <p className="text-sm text-slate-600 mb-3 line-clamp-2 group-hover:text-slate-700">
          {concept.summary}
        </p>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500 group-hover:text-slate-600">{concept.category}</span>
          {concept.related.length > 0 && (
            <span className="text-xs text-slate-400 group-hover:text-brand-400">
              {concept.related.length} related
            </span>
          )}
        </div>

        <div className="text-xs text-slate-400 mt-2 group-hover:text-brand-500 transition">
          → 深入了解
        </div>
      </div>
    </Link>
  );
}
