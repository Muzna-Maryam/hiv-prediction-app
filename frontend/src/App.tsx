import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav";
import PredictPage from "./pages/PredictPage";
import ExplainPage from "./pages/ExplainPage";
import ComparePage from "./pages/ComparePage";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      <Nav />
      <Routes>
        <Route path="/" element={<PredictPage />} />
        <Route path="/explain" element={<ExplainPage />} />
        <Route path="/compare" element={<ComparePage />} />
      </Routes>
    </div>
  );
}
