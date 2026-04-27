export function AnalysisHero({ placeholder }) {
  if (placeholder) {
    return (
      <section className="hero">
        <span className="label">LIMITWEAR VALIDATOR</span>
        <h1>Upload your design to start analysis.</h1>
        <p>We’ll check quality, composition, apparel signal and safety rules.</p>
      </section>
    );
  }

  return (
    <section className="hero">
      <span className="label">ANALYSIS RESULTS</span>
      <h1>Your design analysis is ready.</h1>
      <p>Review the validation result and check what needs attention.</p>
    </section>
  );
}