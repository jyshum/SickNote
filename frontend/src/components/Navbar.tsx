"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-20 w-full px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-6xl items-center justify-between rounded-full border border-slate-200/60 bg-white/90 px-5 py-2.5 shadow-[0_1px_3px_rgba(0,0,0,0.04)] backdrop-blur-xl">
        <Link href="/" className="flex items-center gap-2">
          <Image
            src="/sicknote-logo.png"
            alt="SickNote"
            width={152}
            height={30}
            priority
            className="h-7 w-auto sm:h-8"
          />
        </Link>
        <div className="flex gap-1 text-sm font-medium">
          <Link
            href="/"
            className={`rounded-full px-4 py-2 transition duration-200 ${
              pathname === "/"
                ? "bg-slate-950 text-white"
                : "text-slate-500 hover:text-slate-900"
            }`}
          >
            Analyze
          </Link>
          <Link
            href="/technical"
            className={`rounded-full px-4 py-2 transition duration-200 ${
              pathname === "/technical"
                ? "bg-slate-950 text-white"
                : "text-slate-500 hover:text-slate-900"
            }`}
          >
            How it works
          </Link>
        </div>
      </div>
    </nav>
  );
}
