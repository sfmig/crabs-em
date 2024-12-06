"""Find images with debris in stack."""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.cluster import KMeans

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# input data
input_dir = Path("/home/sminano/swc/project_crab_em/full_res/")

root_dir = input_dir.parent

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# read stack as a collection
coll = io.ImageCollection(str(input_dir) + "/*.tif")
print(len(coll))  # 250


# %%
# read stack as array
image_stack = np.moveaxis(np.array(coll), 0, -1)
print(image_stack.shape)  # x (rows), y (cols), n_images, # (3072, 4096, 250)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define transform of input image
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
resize_target = (308, 406)
transform_image = T.Compose(
    [
        T.ToImage(),  # v2.ToPILImage()
        T.Grayscale(num_output_channels=3),  # cannot be applied to torch tensor
        T.ToDtype(torch.float32, scale=True),
        T.Resize(resize_target),
        # height & width need to be a multiple of patch height & width 14
        # --- so we resize to 3066x4088
        # then downscale 1/10 to fit in memory
    ]
)

print(transform_image(image_stack[:, :, 0]).shape)  # 3, 308, 406

# %%
# Apply transform to all images in the stack
transformed_image_stack = torch.empty(
    (image_stack.shape[-1], 3, resize_target[0], resize_target[1])
)  # batch , channels, height, width

for i in range(image_stack.shape[-1]):
    transformed_image_stack[i, :, :, :] = transform_image(image_stack[:, :, i])

print(transformed_image_stack.shape)  # 308, 406, 250

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute embeddings for each image in the stack using DINO v2
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)


# %%
search_file = coll.files[0]
with torch.no_grad():
    # embedding = dinov2_vits14(load_image(search_file).to(device))
    embeddings_array = dinov2_vits14(transformed_image_stack.to(device))

print(embeddings_array.shape)  # 250, 384


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# K-means clustering of image embeddings
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
    embeddings_array.cpu().numpy()
)

print(kmeans.labels_.shape)

list_labels = np.unique(kmeans.labels_).tolist()
for lbl in list_labels:
    print(sum(kmeans.labels_ == lbl))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect images in clusters with least samples
# we assume cluster 0 (cluster with most labels) is OK images
# and that the rest of clusters represent images w/ debris

# compute idcs per cluster
idcs_in_cluster_i = {}
for lbl in list_labels:
    idcs_in_cluster_i[lbl] = np.where(kmeans.labels_ == lbl)[0]

# inspect images in "debris" cluster
for lbl in list_labels:
    idcs_imgs_in_cluster = idcs_in_cluster_i[lbl]

    # print first 2 images in cluster
    for idx in idcs_imgs_in_cluster[:2]:
        plt.figure()
        plt.imshow(image_stack[:, :, idx], cmap="gray")
        plt.title(f"z-stack index {idx} - label {lbl}")
        plt.show()

    # print last 2
    for idx in idcs_imgs_in_cluster[-2:]:
        plt.figure()
        plt.imshow(image_stack[:, :, idx], cmap="gray")
        plt.title(f"z-stack index {idx} - label {lbl}")
        plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Go thru stack of images and replace images in debris clusters with
# mean of nearest neighbours that are labelled as "not-debris"

# set number of neighbours to compute mean over
n_neighbours = 1

# set output directory
output_dir = (
    root_dir / f"kmeans_and_replace_{n_clusters}clusters_{n_neighbours}nn_DINO/"
)
output_dir.mkdir(exist_ok=True)

for i in range(image_stack.shape[-1]):
    # if image not in cluster 0: replace by mean of nearest neighbours
    if i not in idcs_in_cluster_i[0]:
        idcs_nearest_neighbours = np.argsort(abs(idcs_in_cluster_i[0] - i))[
            :n_neighbours
        ]
        print(f"{i} ---> {idcs_in_cluster_i[0][idcs_nearest_neighbours]}")

        mean_neighbours = np.mean(
            image_stack[
                :,
                :,
                idcs_in_cluster_i[0][idcs_nearest_neighbours],
            ],
            axis=-1,
        )

        io.imsave(
            output_dir / (Path(coll.files[i]).stem.split(".")[0] + "_subs.tif"),
            mean_neighbours,
        )

    # if image in cluster 0: save without modification
    else:
        io.imsave(
            output_dir / (Path(coll.files[i]).stem.split(".")[0] + ".tif"),
            image_stack[:, :, i],
        )

# %%
