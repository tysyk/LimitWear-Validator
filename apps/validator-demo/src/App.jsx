import { useEffect, useState } from "react";
import { Header } from "./components/layout/Header";
import { Footer } from "./components/layout/Footer";
import { AnalysisHero } from "./components/analysis/AnalysisHero";
import { AnalysisResult } from "./components/analysis/AnalysisResult";
import { ImproveBanner } from "./components/analysis/ImproveBanner";
import { GuidelinesPage } from "./components/guidelines/GuidelinesPage";
import { analyzeImage } from "./services/validatorApi";

function App() {
  const [page, setPage] = useState("validator");
  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }, [page]);

  const handleUpload = async (file) => {
    try {
      setLoading(true);
      setError(null);
      setPage("validator");

      setImageUrl(URL.createObjectURL(file));

      const res = await analyzeImage(file);
      const fileSizeMb = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
      const format = file.name.split(".").pop().toUpperCase();

      setResult({
        ...res,
        input: {
          ...res.input,
          fileSize: fileSizeMb,
          format,
        },
      });
    } catch (e) {
      setError("Failed to analyze image");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Header onUpload={handleUpload} />

      <main className="page">
        {page === "guidelines" && (
          <GuidelinesPage onBack={() => setPage("validator")} />
        )}

        {page === "validator" && (
          <>
            {!result && !loading && <AnalysisHero placeholder />}

            {loading && <div className="loading">Analyzing...</div>}

            {error && <div className="error">{error}</div>}

            {result && (
              <>
                <AnalysisHero result={result} />
                <AnalysisResult result={result} imageUrl={imageUrl} />
                <ImproveBanner onOpenGuidelines={() => setPage("guidelines")} />
              </>
            )}
          </>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;