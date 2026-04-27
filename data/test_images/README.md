## LimitWear Validator Test Images

This directory is reserved for a small stable demo/regression image set.

- `valid_apparel/`: clean apparel images expected to return `PASS` or a light `WARN`.
- `branded_apparel/`: apparel with confirmed brand/IP evidence expected to return `FAIL` or `NEED_REVIEW` by policy.
- `non_apparel/`: posters, covers, documents, or unrelated images expected to escalate or fail when ML confidence is high.
- `watermark_cases/`: weak and strong watermark-like examples for false-positive checks.
- `skew_cases/`: document/sketch perspective and skew examples.
- `text_cases/`: text-heavy designs, safe-area text, and OCR/IP examples.
- `low_quality/`: low-resolution or blurry inputs.

Keep the set small and named predictably. The current code is prepared so model weights can be replaced later without changing this structure.
