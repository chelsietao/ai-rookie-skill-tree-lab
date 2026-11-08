export type ContentStatus = "glossary" | "draft" | "reviewed" | "published";
export type ContentLevel = "Glossary" | "Reference" | "Core focus";
export type EdgeType =
  | "prerequisite"
  | "related_to"
  | "used_in"
  | "implements"
  | "optimized_by"
  | "alternative_to"
  | "part_of";

export type ConceptSource = [string, string];  // [title, url]

export interface Concept {
  id: string;
  title: string;
  zh: string;
  category: string;
  subcategory?: string;
  level: ContentLevel;
  summary: string;
  intuition?: string;
  mechanism?: string[];
  diagram?: string[];
  code?: string;
  caseTitle?: string;
  case?: string;
  related: string[];
  sources: ConceptSource[];
  status: ContentStatus;
  lastVerified?: string;
}

export interface ConceptEdge {
  source: string;
  target: string;
  type: EdgeType;
}

export interface CareerPath {
  id: string;
  title: string;
  priority: number;
  focus?: "recommended";
  summary: string;
  markets: {
    singapore: string;
    taiwan: string;
  };
  skills: string[];
  conceptIds: string[];
  roles: string[];
  tradeoffs?: string;
  assessment?: string;
}

export interface GraphNode {
  id: string;
  label: string;
  category: string;
  level: ContentLevel;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
}
