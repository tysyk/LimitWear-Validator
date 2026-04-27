import { useState, useEffect } from "react";

export function ImagePreview({ imageUrl }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "auto";
  }, [open]);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  return (
    <>
      <div className="imageCard">
        <img src={imageUrl} alt="design" />
        <button className="viewBtn" onClick={() => setOpen(true)}>
          View full image ↗
        </button>
      </div>

      {open && (
        <div className="modal" onClick={() => setOpen(false)}>
          <img
            src={imageUrl}
            alt="full"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}