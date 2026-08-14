import { useState } from "react";
import { explain } from "../api/client";
import PatientForm from "../components/PatientForm";
import RiskGauge from "../components/RiskGauge";
import ContributionBars from "../components/ContributionBars";
import type { PatientFeatures, ExplanationResponse } from "../types";

export default function ExplainPage() {
  const [result, setResult] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(features: PatientFeatures) {
    setLoading(true);
    setError(null);
    try {
      const res = await explain(features);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Explanation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "32px 24px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 24 }}>
        <div className="card">
          <h2>Patient data</h2>
          <p>
            Same inputs as the Predict page - this one runs the full SHAP breakdown across
            all 21 features instead of just the top 5.
          </p>
          <PatientForm onSubmit={handleSubmit} submitting={loading} submitLabel="Explain" />
        </div>

        <div className="card">
          <h2>How this prediction was reached</h2>
          {!result && !error && <p>Submit patient data to see the full explanation.</p>}
          {error && <p style={{ color: "var(--risk-high)" }}>{error}</p>}
          {result && (
            <>
              <RiskGauge probability={result.probability_infected} label={result.predicted_label} />

              <div style={{ margin: "20px 0", padding: "12px 14px", background: "var(--surface-sunken)", borderRadius: 8, fontSize: 13 }}>
                <p style={{ margin: 0 }}>
                  <strong>Baseline: </strong>
                  <span className="mono">{(result.base_value * 100).toFixed(1)}%</span> is the
                  model's average predicted probability across a sample of training patients, 
                  the starting point before this patient's specific values are taken into
                  account. Each feature below either pushes the prediction up from that baseline
                  (toward infected) or down (toward not infected). The sum of the baseline plus
                  every contribution below equals the final probability shown above.
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {result && (
        <div className="card" style={{ marginTop: 24 }}>
          <h2>All 21 feature contributions</h2>
          <p>Sorted by strength of influence on this specific prediction, largest first.</p>
          <ContributionBars features={result.all_features} />
        </div>
      )}
    </div>
  );
}
