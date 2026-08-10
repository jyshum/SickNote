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
          <span
            aria-hidden
            className="mx-1 hidden h-5 w-px bg-slate-200 sm:block"
          />
          <a
            href="https://github.com/jyshum/SickNote"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="View SickNote on GitHub"
            title="View source on GitHub"
            className="group flex items-center gap-2 rounded-full px-3 py-2 text-slate-500 transition duration-200 hover:bg-slate-950 hover:text-white"
          >
            <svg
              viewBox="0 0 16 16"
              aria-hidden
              className="h-[18px] w-[18px] transition duration-200 group-hover:rotate-[8deg] group-hover:scale-110"
              fill="currentColor"
            >
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
            </svg>
            <span className="hidden lg:inline">Source</span>
          </a>
        </div>
      </div>
    </nav>
  );
}
