import type { FeatureContribution } from "../types";

// Colors reuse the same risk-gradient tokens as RiskGauge: a feature
// pushing the prediction toward "infected" is colored with the same
// hue as the high end of the gauge, and vice versa. The color system
// is shared across the whole app because it's tied to one meaning
// (direction of risk), not to which component happens to render it.
export default function ContributionBars({
  features,
}: {
  features: FeatureContribution[];
}) {
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.contribution)), 0.0001);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {features.map((f) => {
        const widthPct = (Math.abs(f.contribution) / maxAbs) * 50;
        const positive = f.contribution > 0;
        const color = positive ? "var(--risk-high)" : "var(--risk-low)";

        return (
          <div
            key={f.feature}
            style={{
              display: "grid",
              gridTemplateColumns: "110px 1fr 74px",
              alignItems: "center",
              gap: 10,
            }}
          >
            <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{f.feature}</span>
            <div
              style={{
                position: "relative",
                height: 16,
                background: "var(--surface-sunken)",
                borderRadius: 4,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  bottom: 0,
                  width: 1,
                  background: "var(--border-strong)",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  bottom: 0,
                  borderRadius: 3,
                  background: color,
                  ...(positive
                    ? { left: "50%", width: `${widthPct}%` }
                    : { right: "50%", width: `${widthPct}%` }),
                }}
              />
            </div>
            <span className="mono" style={{ fontSize: 12, textAlign: "right", color }}>
              {positive ? "+" : ""}
              {f.contribution.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
