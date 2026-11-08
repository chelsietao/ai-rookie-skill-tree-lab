import type { Concept } from "@/types";

export function searchConcepts(
  concepts: Concept[],
  query: string
): Concept[] {
  const lowerQuery = query.trim().toLowerCase();
  if (!lowerQuery) return concepts;

  return concepts.filter((concept) => {
    const titleMatch = concept.title.toLowerCase().includes(lowerQuery);
    const zhMatch = concept.zh.toLowerCase().includes(lowerQuery);
    const summaryMatch = concept.summary.toLowerCase().includes(lowerQuery);
    const categoryMatch = concept.category.toLowerCase().includes(lowerQuery);

    return titleMatch || zhMatch || summaryMatch || categoryMatch;
  });
}

export function getCategories(concepts: Concept[]): string[] {
  const categories = new Set<string>();
  concepts.forEach((c) => categories.add(c.category));
  return Array.from(categories).sort();
}

export function filterByCategory(
  concepts: Concept[],
  category: string
): Concept[] {
  if (!category) return concepts;
  return concepts.filter((c) => c.category === category);
}

export function filterByLevel(
  concepts: Concept[],
  level: string
): Concept[] {
  if (!level) return concepts;
  return concepts.filter((c) => c.level === level);
}

export function getRelatedConcepts(
  concept: Concept,
  allConcepts: Concept[]
): Concept[] {
  return concept.related
    .map((id) => allConcepts.find((c) => c.id === id))
    .filter((c): c is Concept => c !== undefined);
}
