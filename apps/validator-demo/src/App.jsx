import { useState } from "react";
import { analyzeImage } from "./services/validatorApi";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (e) {
      alert("Error");
    } finally {
      setLoading(false);
    }
  };

  const copyJson = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
  };

  const verdictClass = result?.verdict
    ? `verdict-${result.verdict.toLowerCase()}`
    : "";

  return (
    <div className="wrapper">
      <div className="container">
        <div className="title">VALIDATOR</div>
        <div className="subtitle">Check your design before drop</div>

        <div className="upload-box">
          <label className="upload-label">
            <input type="file" accept="image/*" onChange={handleFile} />
            <span className="upload-span">CHOOSE FILE</span>
          </label>

          {file && <div className="file-name">{file.name}</div>}
        </div>

        {preview && (
          <div className="preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        <button className="button" onClick={handleAnalyze} disabled={!file || loading}>
          {loading ? "Analyzing..." : "ANALYZE"}
        </button>

        {result && (
          <div className="result">
            <div className={`verdict ${verdictClass}`}>
              {result.verdict}
            </div>

            <div className="score">Score: {result.score} / 100</div>
            <div className="score">Scene: {result.scene?.type || "unknown"}</div>

            <ul className="explain">
              {result.explain?.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>

            <div className="json-block">
              <div className="json-header">
                <span>RAW JSON</span>
                <button onClick={copyJson} className="copy-btn">
                  COPY
                </button>
              </div>

              <pre className="json-content">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;