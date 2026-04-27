export function Header({ onUpload }) {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <header className="header">
      <div className="brand">
        <span>LIMIT</span>
        <strong>WEAR</strong>
      </div>

      <nav className="nav">
        <a>DROPS</a>
        <a>SHOP</a>
        <a>DESIGNERS</a>
        <a>ABOUT</a>
      </nav>

      <div className="headerActions">
        <label className="uploadBtn">
          Upload Design
          <input type="file" hidden onChange={handleFileChange} />
        </label>

        <button className="menuBtn">☰</button>
      </div>
    </header>
  );
}