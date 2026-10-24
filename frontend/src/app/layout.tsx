import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Two-Tower Video Recommender",
  description:
    "A context-aware two-tower recommender built on NVIDIA Merlin, with cold-start handling and a real ablation study.",
};

const NAV_LINKS = [
  { href: "/", label: "Recommender" },
  { href: "/cold-start", label: "Cold Start" },
  { href: "/architecture", label: "Architecture" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-neutral-950 text-neutral-100">
        <header className="border-b border-neutral-800">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <Link href="/" className="font-semibold tracking-tight text-neutral-50">
              Two-Tower Recommender
            </Link>
            <nav className="flex gap-6 text-sm text-neutral-400">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="transition-colors hover:text-neutral-50"
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-neutral-800 py-6 text-center text-xs text-neutral-500">
          Built on{" "}
          <a
            className="underline hover:text-neutral-300"
            href="https://developer.nvidia.com/merlin"
          >
            NVIDIA Merlin
          </a>{" "}
          · inspired by Joyn&apos;s Merlin writeup, see{" "}
          <Link className="underline hover:text-neutral-300" href="/architecture">
            Architecture
          </Link>{" "}
          for credits
        </footer>
      </body>
    </html>
  );
}
