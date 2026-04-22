## Demo Cases

Use this folder for a small, predictable demo set when preparing diploma screenshots or manual API demos.

- `pass/`: clean apparel submissions that should pass or only warn lightly.
- `warn/`: apparel submissions with minor layout issues that should not hard-fail.
- `review/`: ambiguous or non-apparel cases that should escalate to `NEED_REVIEW`.
- `fail/`: clearly blocked cases such as QR codes, exact IP matches, or moderation failures.

Recommended minimum set for a demo:

- 1 normal apparel image
- 1 apparel image with layout/text issue
- 1 non-apparel or poster-like image
- 1 blocked IP/QR example

Keep the set small and stable so the same assets can be reused in screenshots, API demos, and regression checks.
