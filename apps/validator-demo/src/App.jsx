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

  const verdictClass =
    result?.verdict === "PASS"
      ? "pass"
      : result?.verdict === "WARN"
        ? "warn"
        : "fail";

  return (
    <div className="wrapper">
      <div className="container">
        <div className="title">VALIDATOR</div>
        <div className="subtitle">
          Check your design before drop
        </div>

        <div className="upload-box">
          <label className="upload-label">
            <input type="file" onChange={handleFile} />
            <span className="upload-span">CHOOSE FILE</span>
          </label>

          {file && <div className="file-name">{file.name}</div>}
        </div>

        {preview && (
          <div className="preview">
            <img src={preview} />
          </div>
        )}

        <button className="button" onClick={handleAnalyze}>
          {loading ? "Analyzing..." : "ANALYZE"}
        </button>

        {result && (
          <div className="result">
            <div className={`verdict ${verdictClass}`}>
              Verdict: {result.verdict}
            </div>

            <div className="score">
              Score: {result.score} / 100
            </div>

            <div className="score">
              Scene: {result.scene?.type}
            </div>

            <ul className="explain">
              {result.explain?.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>

            <details className="details">
              <summary>Show details</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;