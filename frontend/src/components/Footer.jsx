export default function Footer() {
  return (
    <footer className="border-t border-slate-100 bg-white">
      <div className="mx-auto max-w-6xl px-4 py-6 text-sm text-slate-500 flex flex-col sm:flex-row gap-2 sm:items-center sm:justify-between">
        <div>© {new Date().getFullYear()} Pneumonia Detector (prototype)</div>
        <div className="text-slate-400">
          Not for clinical use. Always consult a qualified professional.
        </div>
      </div>
    </footer>
  );
}

