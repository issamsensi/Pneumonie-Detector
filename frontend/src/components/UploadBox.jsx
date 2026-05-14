import { useEffect, useRef, useState } from 'react';

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export default function UploadBox({
  file,
  onFileChange,
  onAnalyze,
  analyzing,
  disabled,
}) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pickFile = () => inputRef.current?.click();

  const handleFiles = (files) => {
    const f = files?.[0];
    if (!f) return;
    onFileChange(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (disabled) return;
    handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    setDragActive(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-soft p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-slate-900 font-semibold">Upload chest X-ray</div>
          <p className="text-sm text-slate-500 mt-1">
            Drag and drop an image here, or choose a file (JPG/PNG).
          </p>
        </div>
        <button
          type="button"
          onClick={pickFile}
          disabled={disabled}
          className="shrink-0 inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-semibold
            bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Choose file
        </button>
      </div>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !disabled && pickFile()}
        className={[
          'mt-4 rounded-2xl border-2 border-dashed p-4 transition cursor-pointer',
          dragActive ? 'border-medical-400 bg-medical-50' : 'border-slate-200 bg-slate-50',
          disabled ? 'opacity-60 cursor-not-allowed' : '',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={disabled}
        />

        {!file ? (
          <div className="py-10 text-center">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-gradient-to-br from-medical-500 to-brand-500 grid place-items-center shadow-soft">
              <span className="text-white font-semibold">XR</span>
            </div>
            <div className="mt-3 text-sm text-slate-600">
              Drop your X-ray image here
            </div>
            <div className="mt-1 text-xs text-slate-500">
              The image never leaves your machine except to be sent to your local backend for inference.
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
            <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Uploaded chest X-ray preview"
                  className="w-full h-[280px] object-contain bg-white"
                />
              ) : (
                <div className="h-[280px] grid place-items-center text-slate-500 text-sm">
                  Preview unavailable
                </div>
              )}
            </div>

            <div className="flex flex-col gap-3">
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="text-xs text-slate-500">Selected file</div>
                <div className="mt-1 text-sm font-semibold text-slate-900 break-words">
                  {file.name}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {formatBytes(file.size)} · {file.type || 'unknown type'}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onAnalyze();
                  }}
                  disabled={disabled || analyzing}
                  className="flex-1 inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold
                    bg-gradient-to-r from-medical-600 to-brand-600 text-white hover:from-medical-500 hover:to-brand-500
                    disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {analyzing ? 'Analyzing…' : 'Analyze'}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onFileChange(null);
                  }}
                  disabled={disabled || analyzing}
                  className="inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold
                    bg-white border border-slate-200 text-slate-700 hover:bg-slate-50
                    disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  Clear
                </button>
              </div>

              <div className="text-xs text-slate-500">
                Tip: Use a frontal chest X-ray. Results depend on training data and image quality.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

