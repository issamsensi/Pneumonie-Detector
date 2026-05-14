import { useMemo, useState } from 'react';
import Disclaimer from './components/Disclaimer.jsx';
import Footer from './components/Footer.jsx';
import Header from './components/Header.jsx';
import ResultCard from './components/ResultCard.jsx';
import UploadBox from './components/UploadBox.jsx';
import { getApiBaseUrl, predictPneumonia } from './lib/api.js';

function App() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  const onAnalyze = async () => {
    if (!file || analyzing) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictPneumonia(file);
      setResult(data);
    } catch (e) {
      setError(e?.message || 'Prediction failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen app-bg flex flex-col">
      <Header />

      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 flex flex-col gap-6">
              <div className="rounded-2xl border border-slate-200 bg-white shadow-soft p-6">
                <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
                  <div>
                    <div className="inline-flex items-center gap-2 rounded-full bg-medical-50 text-medical-700 border border-medical-100 px-3 py-1 text-xs font-semibold">
                      AI-assisted image classification
                    </div>
                    <h1 className="mt-3 text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
                      Pneumonia detection from chest X-rays
                    </h1>
                    <p className="mt-2 text-slate-600 text-sm">
                      Upload an X-ray image to get a prototype prediction (NORMAL vs PNEUMONIA) with a confidence score.
                    </p>
                  </div>
                  <div className="text-xs text-slate-500">
                    API: <span className="font-mono text-slate-700">{apiBaseUrl}</span>
                  </div>
                </div>
              </div>

              <UploadBox
                file={file}
                onFileChange={(f) => {
                  setFile(f);
                  setResult(null);
                  setError(null);
                }}
                onAnalyze={onAnalyze}
                analyzing={analyzing}
                disabled={false}
              />

              <ResultCard result={result} error={error} />
            </div>

            <aside className="flex flex-col gap-6">
              <Disclaimer />

              <div className="rounded-2xl border border-slate-200 bg-white shadow-soft p-5">
                <div className="font-semibold text-slate-900">How it works</div>
                <ul className="mt-3 space-y-2 text-sm text-slate-600">
                  <li className="flex gap-2">
                    <span className="mt-0.5 h-5 w-5 rounded-full bg-slate-900 text-white grid place-items-center text-xs">
                      1
                    </span>
                    Upload a chest X-ray image (JPG/PNG).
                  </li>
                  <li className="flex gap-2">
                    <span className="mt-0.5 h-5 w-5 rounded-full bg-slate-900 text-white grid place-items-center text-xs">
                      2
                    </span>
                    The backend resizes to 224×224, converts to RGB, and applies DenseNet preprocessing.
                  </li>
                  <li className="flex gap-2">
                    <span className="mt-0.5 h-5 w-5 rounded-full bg-slate-900 text-white grid place-items-center text-xs">
                      3
                    </span>
                    A sigmoid output gives pneumonia probability; ≥ 0.5 maps to PNEUMONIA.
                  </li>
                  <li className="flex gap-2">
                    <span className="mt-0.5 h-5 w-5 rounded-full bg-slate-900 text-white grid place-items-center text-xs">
                      4
                    </span>
                    Grad-CAM highlights the image region that contributed most to the pneumonia score.
                  </li>
                </ul>
              </div>

              <div className="rounded-2xl border border-slate-200 bg-white shadow-soft p-5">
                <div className="font-semibold text-slate-900">Notes</div>
                <p className="mt-2 text-sm text-slate-600">
                  This demo is intentionally conservative: it shows probabilities and warns that predictions can be wrong.
                </p>
              </div>
            </aside>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
