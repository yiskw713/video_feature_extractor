import pandas as pd
import torch


def get_n_samples(csv, n_classes):
    """
    count the number of samples per class using a csv file.
    """
    df = pd.read_csv(csv)

    nums = [0 for i in range(n_classes)]
    for i in range(len(df)):
        cls_id = df.iloc[i]['cls_id']
        nums[cls_id] += 1

    return nums


def get_class_weight(csv, n_classes):
    """
    Class weight for CrossEntropy in the provided dataset by Softbank
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """

    nums = get_n_samples(csv, n_classes)

    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency

    return class_weight
