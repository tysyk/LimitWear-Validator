def check(image_info, ocr, profile):
    max_blocks = profile["max_text_blocks"]
    max_words = profile["max_words"]

    block_count = len(ocr)
    word_count = sum(len(item["text"].split()) for item in ocr)

    if block_count > max_blocks or word_count > max_words:
        return ([{
            "ruleId": "TOO_MUCH_TEXT",
            "title": "Забагато тексту",
            "severity": "MED",
            "message": f"Блоків: {block_count} (max {max_blocks}), слів: {word_count} (max {max_words}).",
            "bbox": None
        }], 15)

    return ([], 0)
