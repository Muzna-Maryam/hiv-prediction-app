import type {
  PatientFeatures,
  PredictionResponse,
  ExplanationResponse,
  ModelRun,
} from "../types";

// Requests go through /api, which vite.config.ts proxies to FastAPI at
// :8000 during dev. In production this would instead be an env var
// (VITE_API_BASE_URL) pointed at wherever the API is deployed - not
// wired up yet since that's a Phase 5 (deploy) concern, not a frontend
// one, but worth remembering this is the one line that changes.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function predict(features: PatientFeatures): Promise<PredictionResponse> {
  return postJSON<PredictionResponse>("/predict", features);
}

export function explain(features: PatientFeatures): Promise<ExplanationResponse> {
  return postJSON<ExplanationResponse>("/explain", features);
}

export async function getModelComparison(): Promise<ModelRun[]> {
  const res = await fetch(`${BASE_URL}/models/comparison`);
  if (!res.ok) throw new Error(`${res.status}: failed to load model comparison`);
  return res.json();
}
