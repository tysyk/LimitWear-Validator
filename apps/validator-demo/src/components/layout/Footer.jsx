export function Footer() {
  return (
    <footer className="footer">
      <div className="footerGrid">
        <div>
          <div className="footerLogo"><span>LIMITWEAR</span></div>
          <p>Streetwear drops. Limited designs.<br />Maximum attitude.</p>

          <div className="socials">
            <span>◎</span>
            <span>♪</span>
            <span>𝕏</span>
            <span>☻</span>
          </div>
        </div>

        <div>
          <h4>SHOP</h4>
          <p>All Drops</p>
          <p>T-Shirts</p>
          <p>Hoodies</p>
          <p>Pants</p>
          <p>Accessories</p>
        </div>

        <div>
          <h4>INFO</h4>
          <p>About Us</p>
          <p>Shipping</p>
          <p>Returns</p>
          <p>Size Guide</p>
          <p>FAQ</p>
        </div>

        <div>
          <h4>SUPPORT</h4>
          <p>Contact Us</p>
          <p>Help Center</p>
          <p>Terms of Service</p>
          <p>Privacy Policy</p>
        </div>

        <div>
          <h4>STAY UPDATED</h4>
          <p>Get exclusive drops and updates.</p>

          <div className="emailBox">
            <input placeholder="Enter your email" />
            <button>→</button>
          </div>
        </div>
      </div>

      <p className="copyright">© 2026 LIMITWEAR. All rights reserved.</p>
    </footer>
  );
}