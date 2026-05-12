export function ImproveBanner({ onOpenGuidelines }) {
  return (
    <section className="improve">
      <div className="improveIcon">♡</div>

      <div className="improveText">
        <h3>Want to improve your score?</h3>
        <p>Check our design guidelines to create the perfect print.</p>
      </div>

      <button onClick={onOpenGuidelines}>View guidelines</button>
    </section>
  );
}