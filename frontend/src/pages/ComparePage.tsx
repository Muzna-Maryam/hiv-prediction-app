import { useEffect, useState } from "react";
import { getModelComparison } from "../api/client";
import type { ModelRun } from "../types";

export default function ComparePage() {
  const [runs, setRuns] = useState<ModelRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelComparison()
      .then(setRuns)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, []);

  const bestAccuracy = runs && runs.length > 0 ? runs[0].accuracy : 0;

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "32px 24px" }}>
      <div className="card">
        <h2>Model comparison</h2>
        <p>
          Latest tuned run per model type, pulled live from MLflow's tracking store. 
          Note that this reflects whichever training run was most recently executed with
          `python -m app.train`, not a fixed snapshot.
        </p>

        {error && <p style={{ color: "var(--risk-high)" }}>{error}</p>}
        {!runs && !error && <p>Loading...</p>}

        {runs && (
          <table className="data">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Recall (infected)</th>
                <th>Best params</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.model_name}>
                  <td style={{ fontFamily: "var(--font-sans)", fontWeight: run.accuracy === bestAccuracy ? 600 : 400 }}>
                    {run.model_name}
                    {run.accuracy === bestAccuracy && (
                      <span
                        style={{
                          marginLeft: 8,
                          fontSize: 11,
                          fontFamily: "var(--font-sans)",
                          fontWeight: 500,
                          color: "var(--accent-strong)",
                          background: "var(--accent-bg)",
                          padding: "2px 8px",
                          borderRadius: 6,
                        }}
                      >
                        deployed
                      </span>
                    )}
                  </td>
                  <td>{(run.accuracy * 100).toFixed(1)}%</td>
                  <td>{(run.recall_infected * 100).toFixed(1)}%</td>
                  <td style={{ fontSize: 11, color: "var(--text-secondary)" }}>
                    {Object.entries(run.params)
                      .map(([k, v]) => `${k.replace("clf__", "")}=${v}`)
                      .join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
