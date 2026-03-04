from rules.checks.size_rule import check as size_check
from rules.checks.safe_area_text_rule import check as safe_area_text_check
from rules.checks.too_much_text_rule import check as too_much_text_check

def run_rules(image_info, ocr, profile):
    score = 100
    violations = []

    checks = [size_check, safe_area_text_check, too_much_text_check]

    for fn in checks:
        v, penalty = fn(image_info, ocr, profile)
        violations.extend(v)
        score -= penalty

    if score < 0:
        score = 0

    return score, violations
