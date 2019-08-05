def get_mean(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        110.63666788 / norm_value, 103.16065604 / norm_value,
        96.29023126 / norm_value
    ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]
