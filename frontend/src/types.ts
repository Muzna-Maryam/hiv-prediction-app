// Mirrors backend/app/schemas.py. Kept as one small hand-written file
// rather than a codegen step (e.g. openapi-typescript) - the API surface
// is 3 endpoints and unlikely to churn fast enough to justify a codegen
// pipeline for a portfolio project. Worth revisiting if this grows.

export interface PatientFeatures {
  trt: number;
  age: number;
  wtkg: number;
  hemo: number;
  homo: number;
  drugs: number;
  karnof: number;
  oprior: number;
  z30: number;
  preanti: number;
  gender: number;
  str2: number;
  strat: number;
  symptom: number;
  treat: number;
  offtrt: number;
  cd40: number;
  cd420: number;
  cd80: number;
  cd820: number;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
}

export interface PredictionResponse {
  probability_infected: number;
  predicted_label: string;
  top_features: FeatureContribution[];
}

export interface ExplanationResponse {
  probability_infected: number;
  predicted_label: string;
  base_value: number;
  all_features: FeatureContribution[];
}

export interface ModelRun {
  model_name: string;
  accuracy: number;
  recall_infected: number;
  params: Record<string, string>;
}
