// Styled after a clinical lab report's reference-range bar - the kind
// that shows where a test result falls between "normal" and "abnormal."
// It's the closest real-world visual vocabulary to what this component
// actually is: one probability, placed on a colored risk scale.

function riskColor(p: number): string {
  if (p < 0.33) return "var(--risk-low)";
  if (p < 0.66) return "var(--risk-mid)";
  return "var(--risk-high)";
}

export default function RiskGauge({
  probability,
  label,
}: {
  probability: number;
  label: string;
}) {
  const pct = Math.round(probability * 100);
  const color = riskColor(probability);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 14 }}>
        <span className="mono" style={{ fontSize: 40, fontWeight: 600, color }}>
          {pct}%
        </span>
        <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>
          predicted probability of infection, classified {label.replace("_", " ")}
        </span>
      </div>

      <div
        style={{
          position: "relative",
          height: 10,
          borderRadius: 6,
          overflow: "hidden",
          background:
            "linear-gradient(to right, var(--risk-low) 0%, var(--risk-low) 33%, var(--risk-mid) 33%, var(--risk-mid) 66%, var(--risk-high) 66%, var(--risk-high) 100%)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `calc(${pct}% - 2px)`,
            top: -4,
            width: 4,
            height: 18,
            background: "var(--text-primary)",
            borderRadius: 2,
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--text-muted)",
          marginTop: 4,
        }}
      >
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
