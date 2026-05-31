import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const outfit = Outfit({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SickNote - Cough Signal Screening",
  description:
    "A cough screening interface that turns short audio samples into confidence-scored model signals.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${outfit.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-[var(--background)] text-[var(--foreground)] font-[family-name:var(--font-body)]">
        <Navbar />
        <div className="min-h-full">{children}</div>
      </body>
    </html>
  );
}
