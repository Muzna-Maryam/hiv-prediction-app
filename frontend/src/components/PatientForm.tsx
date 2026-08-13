import { useState } from "react";
import type { PatientFeatures } from "../types";

// Defaults are plausible clinical trial values, not zero (see earlier
// note: an all-zero request produced a meaningless out-of-distribution
// result). 'time' is intentionally absent - dropped backend-side as a
// leaky feature (see train.py), so it's not part of PatientFeatures at
// all anymore, not just hidden here.
const DEFAULTS: PatientFeatures = {
  trt: 1,
  age: 35,
  wtkg: 70,
  hemo: 0,
  homo: 0,
  drugs: 0,
  karnof: 90,
  oprior: 0,
  z30: 0,
  preanti: 200,
  gender: 1,
  str2: 0,
  strat: 1,
  symptom: 0,
  treat: 1,
  offtrt: 0,
  cd40: 350,
  cd420: 380,
  cd80: 900,
  cd820: 950,
};

// Every categorical/binary field's real value labels, straight from the
// dataset's data dictionary. A raw number input for a field that's only
// ever 0 or 1 makes the person guess or go check documentation - a
// select with the actual labels means the form is self-documenting.
const SELECT_OPTIONS: Partial<Record<keyof PatientFeatures, { value: number; label: string }[]>> = {
  trt: [
    { value: 0, label: "ZDV only" },
    { value: 1, label: "ZDV + ddI" },
    { value: 2, label: "ZDV + Zalcitabine" },
    { value: 3, label: "ddI only" },
  ],
  treat: [
    { value: 0, label: "ZDV only" },
    { value: 1, label: "Other" },
  ],
  strat: [
    { value: 1, label: "Antiretroviral naive" },
    { value: 2, label: ">1 but <=52 weeks prior therapy" },
    { value: 3, label: ">52 weeks prior therapy" },
  ],
  gender: [
    { value: 0, label: "Female" },
    { value: 1, label: "Male" },
  ],
  hemo: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
  homo: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
  drugs: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
  oprior: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
  z30: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
  str2: [
    { value: 0, label: "Naive" },
    { value: 1, label: "Experienced" },
  ],
  symptom: [
    { value: 0, label: "Asymptomatic" },
    { value: 1, label: "Symptomatic" },
  ],
  offtrt: [
    { value: 0, label: "No" },
    { value: 1, label: "Yes" },
  ],
};

// Human-readable labels + units for every field, also straight from the
// data dictionary - replaces raw column names like "wtkg" and "karnof".
const FIELD_LABELS: Record<keyof PatientFeatures, string> = {
  trt: "Treatment arm",
  age: "Age (years)",
  wtkg: "Weight (kg)",
  hemo: "Hemophilia",
  homo: "Homosexual activity",
  drugs: "History of IV drug use",
  karnof: "Karnofsky score (0-100)",
  oprior: "Non-ZDV therapy before trial",
  z30: "ZDV in prior 30 days",
  preanti: "Days of pre-trial antiretroviral therapy",
  gender: "Gender",
  str2: "Antiretroviral history",
  strat: "Antiretroviral history stratum",
  symptom: "Symptomatic",
  treat: "Treatment group",
  offtrt: "Off treatment before 96±5 weeks",
  cd40: "CD4 at baseline",
  cd420: "CD4 at 20±5 weeks",
  cd80: "CD8 at baseline",
  cd820: "CD8 at 20±5 weeks",
};

const GROUPS: { title: string; fields: (keyof PatientFeatures)[] }[] = [
  { title: "Demographics", fields: ["age", "wtkg", "gender"] },
  { title: "Treatment", fields: ["trt", "treat", "offtrt", "oprior", "z30", "strat"] },
  { title: "Medical history", fields: ["hemo", "homo", "drugs", "symptom", "karnof", "str2"] },
  { title: "Lab values", fields: ["preanti", "cd40", "cd420", "cd80", "cd820"] },
];

export default function PatientForm({
  onSubmit,
  submitting,
  submitLabel,
}: {
  onSubmit: (features: PatientFeatures) => void;
  submitting: boolean;
  submitLabel: string;
}) {
  const [values, setValues] = useState<PatientFeatures>(DEFAULTS);

  function update(field: keyof PatientFeatures, raw: string) {
    const num = raw === "" ? 0 : Number(raw);
    setValues((prev) => ({ ...prev, [field]: num }));
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(values);
      }}
    >
      {GROUPS.map((group) => (
        <div key={group.title} style={{ marginBottom: 20 }}>
          <h3
            style={{
              fontSize: 13,
              textTransform: "uppercase",
              letterSpacing: "0.03em",
              color: "var(--text-secondary)",
              marginBottom: 10,
            }}
          >
            {group.title}
          </h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: 12,
            }}
          >
            {group.fields.map((field) => {
              const options = SELECT_OPTIONS[field];
              return (
                <div className="field" key={field}>
                  <label htmlFor={field}>{FIELD_LABELS[field]}</label>
                  {options ? (
                    <select
                      id={field}
                      value={values[field]}
                      onChange={(e) => update(field, e.target.value)}
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: 13,
                        padding: "8px 10px",
                        border: "1px solid var(--border-strong)",
                        borderRadius: 6,
                        background: "var(--surface)",
                        color: "var(--text-primary)",
                      }}
                    >
                      {options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={field}
                      type="number"
                      step="any"
                      value={values[field]}
                      onChange={(e) => update(field, e.target.value)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Running model..." : submitLabel}
      </button>
    </form>
  );
}
