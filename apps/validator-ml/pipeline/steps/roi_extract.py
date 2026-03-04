def run(ctx):
    # Поки що ROI = вся картинка.
    # Далі: для mockup_apparel додамо print-zone/segmentation.
    ctx.roi = {"bbox": [0, 0, ctx.width, ctx.height], "confidence": 1.0}
