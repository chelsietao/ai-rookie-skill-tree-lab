"use client";

import { useMemo } from "react";
import Link from "next/link";
import type { Concept } from "@/types";

interface ConceptGraphProps {
  concepts: Concept[];
}

export default function ConceptGraph({ concepts }: ConceptGraphProps) {
  const categories = useMemo(() => {
    const cats = new Map<string, Concept[]>();
    concepts.forEach((c) => {
      if (!cats.has(c.category)) {
        cats.set(c.category, []);
      }
      cats.get(c.category)!.push(c);
    });
    return cats;
  }, [concepts]);

  return (
    <div className="w-full h-full overflow-auto p-6 bg-gradient-to-br from-slate-50 to-white">
      <div className="flex flex-wrap gap-8">
        {Array.from(categories.entries()).map(([category, cats]) => (
          <div key={category} className="flex-1 min-w-64">
            <h3 className="text-sm font-semibold text-slate-700 mb-4 pb-2 border-b border-slate-200">
              {category}
            </h3>
            <div className="flex flex-col gap-2">
              {cats
                .slice(0, 6)
                .map((c) => (
                  <Link
                    key={c.id}
                    href={`/concepts/${c.id}`}
                    className="text-sm text-brand-600 hover:text-brand-700 hover:underline p-2 rounded hover:bg-brand-50 transition"
                  >
                    {c.title}
                  </Link>
                ))}
              {cats.length > 6 && (
                <div className="text-xs text-slate-500 p-2">
                  +{cats.length - 6} more
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
