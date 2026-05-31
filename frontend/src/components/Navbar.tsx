"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 z-50 w-full px-3 py-3 sm:px-6">
      <div className="mx-auto flex max-w-6xl items-center justify-between rounded-full border border-slate-200/80 bg-white/88 px-4 py-2.5 shadow-[0_18px_60px_-42px_rgba(15,23,42,0.32)] backdrop-blur-2xl sm:px-5">
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
        <div className="flex items-center gap-1 text-sm font-semibold">
          <Link
            href="/"
            className={`rounded-full px-3 py-2 transition duration-200 sm:px-4 ${
              pathname === "/"
                ? "bg-slate-950 text-white"
                : "text-slate-500 hover:text-slate-950"
            }`}
          >
            Analyze
          </Link>
          <Link
            href="/technical"
            className={`rounded-full px-3 py-2 transition duration-200 sm:px-4 ${
              pathname === "/technical"
                ? "bg-slate-950 text-white"
                : "text-slate-500 hover:text-slate-950"
            }`}
          >
            How it works
          </Link>
        </div>
      </div>
    </nav>
  );
}
