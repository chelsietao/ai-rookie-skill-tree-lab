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
      <div className="space-y-6">
        {Array.from(categories.entries()).map(([category, cats]) => (
          <div key={category} className="border-l-4 border-brand-500 pl-6">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">
              {category}
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {cats
                .slice(0, 9)
                .map((c) => (
                  <Link
                    key={c.id}
                    href={`/concepts/${c.id}`}
                    className="text-sm text-brand-600 hover:text-brand-700 hover:underline p-2 rounded hover:bg-brand-50 transition block font-medium"
                  >
                    {c.title}
                  </Link>
                ))}
              {cats.length > 9 && (
                <div className="text-xs text-slate-500 p-2 col-span-1 flex items-center">
                  +{cats.length - 9} more
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
