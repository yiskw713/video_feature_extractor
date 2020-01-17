def get_mean(norm_value=255):
    # mean of imagenet
    return [
        123.675 / norm_value, 116.28 / norm_value,
        103.53 / norm_value
    ]


def get_std(norm_value=255):
    # std of imagenet
    return [
        58.395 / norm_value, 57.12 / norm_value,
        57.375 / norm_value
    ]
