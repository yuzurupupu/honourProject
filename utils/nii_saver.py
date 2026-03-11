import nibabel as nib
import numpy as np


def save_nii(volume,path):

    nii = nib.Nifti1Image(volume,np.eye(4))

    nib.save(nii,path)