from ml.adult_safety.inference_adult_safety import predict_adult_safety


def run(ctx):
    result = predict_adult_safety(ctx.bgr)

    ctx.detections["ml_adult_safety"] = result
    ctx.ml["adult_safety"] = result

    return ctx