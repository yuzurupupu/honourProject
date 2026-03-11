from scipy import linalg
import numpy as np


def calculate_fid(mu1,sigma1,mu2,sigma2):

    diff = mu1 - mu2

    covmean = linalg.sqrtm(sigma1 @ sigma2)

    fid = diff.dot(diff) + np.trace(
        sigma1 + sigma2 - 2*covmean
    )

    return fid