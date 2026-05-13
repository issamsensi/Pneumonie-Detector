const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export function getApiBaseUrl() {
  return import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL;
}

export async function predictPneumonia(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${getApiBaseUrl()}/predict`, {
    method: 'POST',
    body: formData,
  });

  let data;
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    data = await res.json();
  } else {
    data = { detail: await res.text() };
  }

  if (!res.ok) {
    const message =
      (typeof data?.detail === 'string' && data.detail) ||
      'Prediction failed. Please try another image.';
    const err = new Error(message);
    err.status = res.status;
    err.payload = data;
    throw err;
  }

  return data;
}

