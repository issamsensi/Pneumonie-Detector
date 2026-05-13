export default function Header() {
  return (
    <header className="bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60 sticky top-0 z-10 border-b border-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-medical-500 to-brand-500 shadow-soft grid place-items-center">
            <span className="text-white font-semibold">AI</span>
          </div>
          <div>
            <div className="text-slate-900 font-semibold leading-tight">Pneumonia Detector</div>
            <div className="text-slate-500 text-sm">Chest X-ray analysis (prototype)</div>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-xs">
          <span className="rounded-full bg-medical-50 text-medical-700 border border-medical-100 px-2 py-1">
            FastAPI
          </span>
          <span className="rounded-full bg-brand-50 text-brand-700 border border-brand-100 px-2 py-1">
            TensorFlow/Keras
          </span>
          <span className="rounded-full bg-slate-50 text-slate-700 border border-slate-200 px-2 py-1">
            React + Vite
          </span>
        </div>
      </div>
    </header>
  );
}

