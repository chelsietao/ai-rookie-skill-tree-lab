import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center px-6">
      <div className="text-center max-w-md">
        <h1 className="text-6xl font-bold text-slate-900 mb-4">404</h1>
        <p className="text-xl text-slate-600 mb-2">Page not found</p>
        <p className="text-slate-600 mb-8">
          The concept or page you&apos;re looking for doesn&apos;t exist or has moved.
        </p>
        <div className="flex gap-4 justify-center">
          <Link
            href="/"
            className="px-6 py-2 bg-brand-500 text-white rounded-lg hover:bg-brand-600 transition"
          >
            Back to Home
          </Link>
          <Link
            href="/concepts"
            className="px-6 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition"
          >
            All Concepts
          </Link>
        </div>
      </div>
    </div>
  );
}
