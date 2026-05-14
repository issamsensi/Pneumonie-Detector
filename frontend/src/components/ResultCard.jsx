function Badge({ label, tone }) {
  const tones = {
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    red: 'bg-rose-50 text-rose-700 border-rose-200',
    slate: 'bg-slate-50 text-slate-700 border-slate-200',
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium ${tones[tone] || tones.slate}`}>
      {label}
    </span>
  );
}

export default function ResultCard({ result, error }) {
  if (!result && !error) return null;

  if (error) {
    return (
      <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5">
        <div className="flex items-center justify-between gap-3">
          <div className="font-semibold text-rose-900">Couldn’t analyze the image</div>
          <Badge label="Error" tone="red" />
        </div>
        <p className="text-sm text-rose-800 mt-2">{error}</p>
      </div>
    );
  }

  const isPneumonia = result?.label === 'PNEUMONIA';
  const confidencePct = Math.round((result?.confidence ?? 0) * 100);
  const pneumoniaPct = Math.round((result?.pneumonia_probability ?? 0) * 100);
  const gradcam = result?.gradcam;
  const gradcamSrc = gradcam?.overlay_image_base64
    ? `data:image/png;base64,${gradcam.overlay_image_base64}`
    : null;
  const bbox = gradcam?.bbox_normalized;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-soft p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm text-slate-500">Prediction</div>
          <div className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
            {isPneumonia ? 'PNEUMONIA' : 'NORMAL'}
          </div>
        </div>
        <Badge label={isPneumonia ? 'High attention' : 'Low attention'} tone={isPneumonia ? 'red' : 'green'} />
      </div>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-xs text-slate-500">Confidence</div>
          <div className="mt-1 text-xl font-semibold text-slate-900">{confidencePct}%</div>
          <div className="mt-2 h-2 rounded-full bg-white border border-slate-200 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-medical-500 to-brand-500" style={{ width: `${confidencePct}%` }} />
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-xs text-slate-500">Pneumonia probability</div>
          <div className="mt-1 text-xl font-semibold text-slate-900">{pneumoniaPct}%</div>
          <div className="mt-2 text-xs text-slate-500">
            Sigmoid output; ≥ 50% maps to class index 1.
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
        <div className="text-sm font-semibold text-slate-900">Explanation</div>
        <p className="text-sm text-slate-600 mt-1">
          This is a prototype AI classifier trained on chest X-ray images. The result is probabilistic and can be wrong.
          Use it only for learning and demonstrations—not for clinical decisions.
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm font-semibold text-slate-900">Grad-CAM localization</div>
          <Badge label="Experimental" tone="slate" />
        </div>

        {gradcamSrc ? (
          <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-lg border border-slate-200 overflow-hidden bg-white">
              <img
                src={gradcamSrc}
                alt="Grad-CAM heatmap overlay on the uploaded X-ray"
                className="w-full h-[260px] object-contain bg-white"
              />
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs text-slate-500">Most activated area</div>
              <div className="mt-1 text-sm font-semibold text-slate-900">
                {gradcam?.location_text || 'Unavailable'}
              </div>

              {bbox ? (
                <div className="mt-3 text-xs text-slate-600 space-y-1">
                  <div>x_min: {bbox.x_min}</div>
                  <div>y_min: {bbox.y_min}</div>
                  <div>x_max: {bbox.x_max}</div>
                  <div>y_max: {bbox.y_max}</div>
                </div>
              ) : null}

              <p className="mt-3 text-xs text-slate-500">
                Heatmap focuses on features that increased the pneumonia score. This is not a pixel-accurate lesion segmentation.
              </p>
            </div>
          </div>
        ) : (
          <p className="mt-2 text-sm text-slate-600">
            Grad-CAM output is not available for this prediction.
          </p>
        )}
      </div>
    </div>
  );
}

