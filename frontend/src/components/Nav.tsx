import { NavLink } from "react-router-dom";

const LINK_STYLE_BASE: React.CSSProperties = {
  padding: "8px 14px",
  borderRadius: 8,
  fontSize: 14,
  fontWeight: 500,
  textDecoration: "none",
  color: "var(--text-secondary)",
};

export default function Nav() {
  return (
    <nav
      style={{
        display: "flex",
        gap: 4,
        padding: "16px 24px",
        borderBottom: "1px solid var(--border)",
        background: "var(--surface)",
      }}
    >
      <span className="mono" style={{ fontWeight: 600, marginRight: 20, color: "var(--text-primary)" }}>
        HIV outcome predictor
      </span>
      {[
        { to: "/", label: "Predict" },
        { to: "/explain", label: "Explain" },
        { to: "/compare", label: "Compare models" },
      ].map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.to === "/"}
          style={({ isActive }) => ({
            ...LINK_STYLE_BASE,
            background: isActive ? "var(--accent-bg)" : "transparent",
            color: isActive ? "var(--accent-strong)" : "var(--text-secondary)",
          })}
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
