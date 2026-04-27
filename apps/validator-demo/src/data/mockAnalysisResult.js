export const mockAnalysisResult = {
  analysisId: "demo-001",
  profileId: "default",
  score: 82,
  verdict: "WARN",
  imageUrl:
    "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=900&q=80",
  input: {
    width: 1024,
    height: 1260,
    format: "PNG",
    fileSize: "1.2 MB",
  },
  summary: {
    headline: "Your design analysis is ready.",
    description: "Review the results below and see how your design scores.",
  },
  metrics: [
    { label: "Quality", value: "Good", status: "good" },
    { label: "Text", value: "Medium", status: "medium" },
    { label: "Composition", value: "Good", status: "good" },
    { label: "Safety", value: "Good", status: "good" },
  ],
  violations: [
    {
      ruleId: "TEXT_NEAR_EDGE",
      title: "Text near edge",
      message: "Some text elements are too close to the edge.",
      severity: "high",
    },
    {
      ruleId: "LOGO_SIZE",
      title: "Logo size",
      message: "The logo is quite large. Consider reducing it.",
      severity: "medium",
    },
    {
      ruleId: "LOW_CONTRAST",
      title: "Low contrast",
      message: "Some areas may print with low contrast.",
      severity: "low",
    },
  ],
};