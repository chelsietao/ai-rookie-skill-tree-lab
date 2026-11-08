import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Engineering Atlas — Interactive Encyclopedia",
  description:
    "Explore AI Engineering from models to systems. Interactive knowledge graph covering inference optimization, compilers, GPU computing, and distributed systems.",
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='75' font-size='75' fill='%230284c7'>⚡</text></svg>",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-Hant">
      <body className="bg-white text-slate-900">{children}</body>
    </html>
  );
}
