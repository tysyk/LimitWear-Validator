export function GuidelinesPage({ onBack }) {
  return (
    <section className="guidelinesPage">
      <div className="guidelinesInner">
        <div className="guidelinesTopbar">
          <button className="guidelinesBack" onClick={onBack}>
            <span>←</span>
            Back to validator
          </button>
        </div>

        <div className="guidelinesHero">
          <span className="guidelinesEyebrow">Design rules</span>

          <h1>How to prepare your design</h1>

          <p>
            Follow these rules to improve your validation score and avoid manual
            review. The validator checks image quality, brand risks, text,
            safety and possible copyright issues.
          </p>
        </div>

        <div className="guidelinesGrid">
          <article className="guidelineCard">
            <span className="guidelineNumber">01</span>
            <h3>Use a clear image</h3>
            <p>
              Upload a sharp image where the print is easy to see. Avoid blurry,
              pixelated, dark or heavily compressed photos.
            </p>
          </article>

          <article className="guidelineCard">
            <span className="guidelineNumber">02</span>
            <h3>Keep the design visible</h3>
            <p>
              The main print should not be cut off or hidden. If it is a photo
              of clothing, make sure the garment is centered and visible.
            </p>
          </article>

          <article className="guidelineCard">
            <span className="guidelineNumber">03</span>
            <h3>Avoid known brands</h3>
            <p>
              Do not use logos, brand names, slogans or recognizable marks from
              companies such as Nike, Adidas, Gucci, Supreme and others unless
              you own the rights.
            </p>
          </article>

          <article className="guidelineCard">
            <span className="guidelineNumber">04</span>
            <h3>Do not use copyrighted characters</h3>
            <p>
              Avoid characters, franchises or protected content from movies,
              games, anime, cartoons or popular entertainment brands.
            </p>
          </article>

          <article className="guidelineCard">
            <span className="guidelineNumber">05</span>
            <h3>No watermark or stock preview</h3>
            <p>
              Designs with watermark, preview marks or stock image signs may be
              sent to manual review or rejected.
            </p>
          </article>

          <article className="guidelineCard">
            <span className="guidelineNumber">06</span>
            <h3>Keep it safe</h3>
            <p>
              Do not upload adult, hateful, extremist, violent or unsafe
              content. These designs can be blocked automatically.
            </p>
          </article>
        </div>

        <div className="verdictGuide">
          <h2>Validation results</h2>

          <div className="verdictGuideGrid">
            <div className="verdictInfo pass">
              <strong>PASS</strong>
              <p>
                The design passed automatic validation. No critical or
                suspicious risks were found.
              </p>
            </div>

            <div className="verdictInfo review">
              <strong>NEED REVIEW</strong>
              <p>
                The design is not rejected, but an admin should check it
                manually. This can happen because of a possible brand, watermark
                or unclear image.
              </p>
            </div>

            <div className="verdictInfo fail">
              <strong>FAIL</strong>
              <p>
                The design contains a blocking risk and cannot be accepted
                automatically.
              </p>
            </div>
          </div>
        </div>

        <div className="checklistBox">
          <h2>Before uploading, check this</h2>

          <ul>
            <li>The image is sharp and not too small.</li>
            <li>The full design is visible.</li>
            <li>There are no famous logos or brand names.</li>
            <li>There are no copyrighted characters or slogans.</li>
            <li>There are no watermarks or stock preview marks.</li>
            <li>The design does not contain unsafe content.</li>
          </ul>
        </div>
      </div>
    </section>
  );
}