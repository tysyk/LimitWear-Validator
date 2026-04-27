export function ScorePanel({ result }) {
  const score = result?.score ?? 0;
  const verdict = result?.verdict ?? "NEED_REVIEW";
  const input = result?.input || {};
  const violations = result?.violations || [];
  const ruleResults = result?.ruleResults || [];
  const moderation = result?.moderation || {};
  const quality = result?.quality || {};

  const hasTextIssues = violations.some((v) =>
    String(v.ruleId || "").includes("TEXT")
  );

  const hasCompositionIssues = violations.some((v) =>
    ["VISUAL_LOGO_CENTER", "TEXT_NEAR_EDGE", "MESSY_LINES", "HIGH_SKEW"].includes(
      v.ruleId
    )
  );

  const hasSafetyIssues =
    moderation.blocked ||
    moderation.needsReview ||
    violations.some((v) =>
      ["MODERATION", "IP", "BRAND", "WATERMARK"].some((key) =>
        String(v.ruleId || "").includes(key)
      )
    );

  const qualityScore = quality.quality_score ?? 1;
  const hasQualityIssue =
    qualityScore < 0.7 ||
    violations.some((v) =>
      ["LOW_RESOLUTION", "BLURRY_IMAGE"].includes(v.ruleId)
    );

  const metrics = [
    {
      label: "Quality",
      value: hasQualityIssue ? "Check" : "Good",
      status: hasQualityIssue ? "medium" : "good",
      icon: "⌘",
    },
    {
      label: "Text",
      value: hasTextIssues ? "Check" : "Good",
      status: hasTextIssues ? "medium" : "good",
      icon: "T",
    },
    {
      label: "Composition",
      value: hasCompositionIssues ? "Check" : "Good",
      status: hasCompositionIssues ? "medium" : "good",
      icon: "✣",
    },
    {
      label: "Safety",
      value: hasSafetyIssues ? "Risk" : "Good",
      status: hasSafetyIssues ? "medium" : "good",
      icon: "♡",
    },
  ];

  const verdictText = {
    PASS: "Everything looks good",
    WARN: "Some issues found",
    FAIL: "Blocking issues found",
    NEED_REVIEW: "Manual review recommended",
  };

  return (
    <div className="scoreCard">
      <div className="scoreTop">
        <div>
          <div className="scoreLabel">Overall Score</div>
          <div className="scoreValue">
            {score}
            <span>/100</span>
          </div>
          <div className="progress">
            <div className="progressFill" style={{ width: `${score}%` }} />
          </div>
        </div>

        <div className="divider" />

        <div className="verdictBox">
          <div className="scoreLabel">Verdict</div>
          <div className="verdictValue">{verdict.replace("_", " ")}</div>
          <p>{verdictText[verdict] || "Analysis completed"}</p>
        </div>
      </div>

      <div className="fileMetrics">
        <div>
          <span>Resolution</span>
          <strong>
            {input.width || "-"} × {input.height || "-"}
          </strong>
          <small className="goodText">Good</small>
        </div>

        <div>
          <span>File Size</span>
          <strong>{input.fileSize || "-"}</strong>
          <small className="goodText">Good</small>
        </div>

        <div>
          <span>Format</span>
          <strong>{input.format || "PNG"}</strong>
          <small className="goodText">Good</small>
        </div>
      </div>

      <div className="metricRow">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className={`metricItem ${metric.status === "medium" ? "medium" : ""}`}
          >
            <div className="metricIcon">{metric.icon}</div>
            <div className="metricText">
              <strong>{metric.label}</strong>
              <span>{metric.value}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}