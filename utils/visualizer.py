import matplotlib.pyplot as plt


def save_slice(volume,path):

    z = volume.shape[0]//2

    slice = volume[z]

    plt.imshow(slice,cmap="gray")

    plt.axis("off")

    plt.savefig(path)