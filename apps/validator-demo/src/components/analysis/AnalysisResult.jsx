import { ImagePreview } from "./ImagePreview";
import { ScorePanel } from "./ScorePanel";
import { IssueList } from "./IssueList";

export function AnalysisResult({ result, imageUrl }) {
  return (
    <section className="analysisGrid">
      <ImagePreview imageUrl={imageUrl} />
      <ScorePanel result={result} />
      <IssueList issues={result.violations || []} />
    </section>
  );
}