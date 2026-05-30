"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="w-full border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-[var(--green)]" />
          <span className="text-lg font-bold text-slate-900">SickNote</span>
        </Link>
        <div className="flex gap-6 text-sm font-medium">
          <Link
            href="/"
            className={pathname === "/" ? "text-slate-900" : "text-slate-500 hover:text-slate-700"}
          >
            Analyze
          </Link>
          <Link
            href="/technical"
            className={pathname === "/technical" ? "text-slate-900" : "text-slate-500 hover:text-slate-700"}
          >
            How it Works
          </Link>
        </div>
      </div>
    </nav>
  );
}
