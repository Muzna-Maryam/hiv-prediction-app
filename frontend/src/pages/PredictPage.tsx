import { useState } from "react";
import { predict } from "../api/client";
import PatientForm from "../components/PatientForm";
import RiskGauge from "../components/RiskGauge";
import ContributionBars from "../components/ContributionBars";
import type { PatientFeatures, PredictionResponse } from "../types";

export default function PredictPage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(features: PatientFeatures) {
    setLoading(true);
    setError(null);
    try {
      const res = await predict(features);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "32px 24px", display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 24 }}>
      <div className="card">
        <h2>Patient data</h2>
        <p>Enter clinical values to get a predicted infection probability.</p>
        <PatientForm onSubmit={handleSubmit} submitting={loading} submitLabel="Predict" />
      </div>

      <div className="card">
        <h2>Result</h2>
        {!result && !error && (
          <p>Submit patient data to see a prediction.</p>
        )}
        {error && <p style={{ color: "var(--risk-high)" }}>{error}</p>}
        {result && (
          <>
            <RiskGauge probability={result.probability_infected} label={result.predicted_label} />
            <h3 style={{ marginTop: 24, fontSize: 13, textTransform: "uppercase", letterSpacing: "0.03em", color: "var(--text-secondary)" }}>
              Top contributing factors
            </h3>
            <ContributionBars features={result.top_features} />
            <p style={{ marginTop: 16, fontSize: 12 }}>
              Bars pushing right (toward the risk color) increase the predicted probability; bars pushing left decrease it.
              See the Explain page for the full breakdown.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
