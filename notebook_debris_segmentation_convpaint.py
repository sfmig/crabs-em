# %%
# From https://guiwitz.github.io/napari-convpaint/book/convpaint_api.html
#
# In a conda environment, get latest version of napari-convpaint with:
#   `pip install git+https://github.com/guiwitz/napari-convpaint.git@a30a35334dc38495444007e32f03393366792838`


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
from napari_convpaint import conv_paint, conv_paint_param

# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define input data
# image collection
input_dir = Path("/home/sminano/swc/project_crab_em/full_res/")
coll = io.ImageCollection(str(input_dir) + "/*.tif")
print(len(coll))  # 250


# read stack as array
image_stack = np.array(coll)
print(image_stack.shape)  # (250, 3072, 4096) ==> n_images, x (rows), y (cols)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define annotations

annotations_dir = Path("/home/sminano/swc/project_crab_em/masks/tile8_3x4K_mask_tiff")
annotations_coll = io.ImageCollection(str(annotations_dir) + "/*.tif")
print(len(annotations_coll))  # 250

# read annotations as array
annotations_stack = np.array(annotations_coll) + 1.0
# we add +1 to annotations to match the labels in the napari-convpaint library
# Initially: 0 = background, 1 = debris
# Now: 1 = background, 2 = debris, 0 = no annotation
print(annotations_stack.shape)  # (250, 3072, 4096)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot annotations for a sample image

sel_idx = 15
image_array = image_stack[sel_idx, :, :]
annotations_array = annotations_stack[sel_idx, :, :]

plt.figure()
plt.imshow(image_array, cmap="gray")
# if img_right has 3 channels, cmap is ignored by matplotlib
plt.imshow(annotations_array, alpha=0.5, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.title("Ground truth")

print("Number of annotated pixels: ", np.sum(annotations_array > 0))
print("Labels: ", np.unique(annotations_array))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set DINOv2 as the feature extractor
param_dino = conv_paint_param.Param()
param_dino.fe_name = "dinov2_vits14_reg"
param_dino.fe_scalings = [1]
param_dino.fe_order = 0
param_dino.image_downsample = 10  # 8

# create model
model_dino = conv_paint.create_model(param_dino)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set VGG16 as the feature extractor
param_vgg = conv_paint_param.Param()
param_vgg.fe_name = "vgg16"
param_vgg.fe_scalings = [1, 2, 4]
param_vgg.fe_order = 0
param_vgg.fe_layers = [
    "features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
    "features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
]
param_vgg.image_downsample = 15

# create model
model_vgg = conv_paint.create_model(param_vgg)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Images classified with debris
# (from classification notebook)
idcs_images_w_debris = [
    6,
    15,
    21,
    24,
    29,
    40,
    41,
    42,
    58,
    72,
    82,
    84,
    86,
    104,
    105,
    109,
    125,
    156,
    157,
    165,
    167,
    170,
    176,
    178,
    181,
    186,
    200,
    206,
    207,
    210,
    227,
    228,
    231,
    242,
    243,
    246,
]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define train / test split only on images with debris
n_training_samples = math.ceil(0.8 * len(idcs_images_w_debris))
n_test_samples = len(idcs_images_w_debris) - n_training_samples

rng = np.random.default_rng(42)
train_idcs = rng.choice(idcs_images_w_debris, n_training_samples, replace=False)
test_idcs = np.setdiff1d(idcs_images_w_debris, train_idcs)

print(train_idcs)
print(test_idcs)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%§
# Train pixel classifier on multiple debris images
# TODO: normalise image stack?

model = model_vgg
param = param_vgg

all_features, all_targets = [], []
for idx in train_idcs:
    features, targets = conv_paint.get_features_current_layers(
        image_stack[idx, :, :],
        annotations_stack[idx, :, :],
        model=model,
        param=param,
    )
    all_features.append(features)
    all_targets.append(targets)

features_array = np.concatenate(all_features, axis=0)
targets_array = np.concatenate(all_targets, axis=0)

# %%
# train classifier
random_forest = conv_paint.train_classifier(features_array, targets_array)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on another image (e.g. 15) ---> this throws an error with DINO
# ValueError: cannot reshape array of size 638 into shape (3,3)

# for sanity check: evaluate on same image as training
# (that should be the best performance)
for image_idx in test_idcs:
    predicted_image = model.predict_image(
        image=image_stack[image_idx, :, :],
        classifier=random_forest,
        param=param,
    )

    plt.figure()
    plt.imshow(image_stack[image_idx, :, :], cmap="gray")
    plt.imshow(predicted_image, alpha=0.5, cmap="viridis", interpolation="nearest")
    plt.title(f"Prediction sample: {image_idx} ")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect error in reshaping

# We go through the lines in `predict_image` at
# https://github.com/guiwitz/napari-convpaint/blob/a30a35334dc38495444007e32f03393366792838/src/napari_convpaint/conv_paint_dino.py#L199
# to find the issue

# for sanity check: evaluate on same image as training
# (that should be the best performance)
for image_idx in test_idcs:
    # inputs
    image = image_stack[image_idx, :, :]
    classifier = random_forest
    param = param

    # add padding to the image if required
    padding = param.fe_padding
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)  # if 2d, prepend an additional dimension
    image = np.pad(
        image, ((0, 0), (padding, padding), (padding, padding)), mode="reflect"
    )

    # compute feature vectors for each block of pixels (?) in downscaled image
    features = model.get_features_scaled(image, param, return_patches=True)
    nb_features = features.shape[0]  # [nb_features, width, height]
    w_patch = features.shape[-2]
    h_path = features.shape[-1]
    # Last two lines are originally:
    # w_patch = np.ceil(features.shape[-2] / param.image_downsample).astype(int)
    # h_path = np.ceil(features.shape[-1] / param.image_downsample).astype(int)

    print(features.shape)  # [nb_features, width, height]
    print(w_patch)
    print(h_path)
    print(w_patch * h_path)

    # predict label for  features
    # move features to last dimension
    # (we need to make it 1d-array for RF = random forest)
    features = np.moveaxis(features, 0, -1)  # [width, height, nb_features]
    features = np.reshape(features, (-1, nb_features))  # [width*height, nb_features]
    print(features.shape)

    # predict using trained random forest classifier
    predictions = classifier.predict(pd.DataFrame(features))
    print(predictions.shape)  # 638

    # reshape predictions to feature shape
    predicted_image = np.reshape(predictions, [w_patch, h_path])
    print(predicted_image.shape)  # (29, 22)

    # resize predicted image to original image size
    predicted_image = skimage.transform.resize(
        image=predicted_image,
        output_shape=(image.shape[-2], image.shape[-1]),
        preserve_range=True,
        order=param.fe_order,
    ).astype(np.uint8)
    print(predicted_image.shape)  # (4096, 3072)

    # apply padding if required
    if padding > 0:
        predicted_image = predicted_image[padding:-padding, padding:-padding]

    ##
    plt.figure()
    plt.imshow(image_stack[image_idx, :, :], cmap="gray")
    plt.imshow(predicted_image, alpha=0.5, cmap="viridis", interpolation="nearest")
    plt.title(f"Prediction sample: {image_idx} ")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TODO next:
# - Understand why we divide by downsampling factor and fix error
# - Do we need to train & evaluate images with debris only? -- seems like a good idea
# - Try downsampling less?
# - Try concatenating features from DINO and VGG16? (VGG better spatial resolution?)
#   - VGG is grainy ... try more layers?
# - Compute accuracy metric & compare feature extractors
# - Train one classifier per z-stack?
