# %%
# From https://guiwitz.github.io/napari-convpaint/book/convpaint_api.html
# pip install git+https://github.com/guiwitz/napari-convpaint.git@a30a35334dc38495444007e32f03393366792838


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
import numpy as np
import skimage
from napari_convpaint import conv_paint, conv_paint_param

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set DINOv2 as the feature extractor
param = conv_paint_param.Param()
param.fe_name = "dinov2_vits14_reg"
param.fe_scalings = [1]
param.fe_order = 0
param.image_downsample = 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define input data
img_left, img_right, disparity = skimage.data.stereo_motorcycle()

plt.figure()
plt.imshow(img_left)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define annotations
annotations = np.zeros(img_left[:, :, 0].shape)
# foreground [y,x]
annotations[50:100, 50:100] = 1
annotations[450:500, 500:550] = 1
# background [x,y]
annotations[200:250, 400:450] = 2
annotations[300:350, 200:400] = 2

plt.figure()
plt.imshow(annotations, interpolation="nearest")


print("Number of annotated pixels: ", np.sum(annotations > 0))
print("Pixels annotated as foreground: ", np.sum(annotations == 1))
print("Pixels annotated as background: ", np.sum(annotations == 2))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create model
model = conv_paint.create_model(param)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# train pixel classifier
features, targets = conv_paint.get_features_current_layers(
    np.moveaxis(img_left, -1, 0), annotations, model=model, param=param
)

random_forest = conv_paint.train_classifier(features, targets)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# run inference
prediction = model.predict_image(
    image=np.moveaxis(img_right, -1, 0),  # move channels to the front
    classifier=random_forest,
    param=param,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot inference image and prediction
# Plot the prediction
plt.figure()
plt.imshow(img_right)  # , cmap='gray')
plt.title("Input image")

plt.figure()
plt.imshow(img_left)  # , cmap='gray')
plt.title("Target image")

plt.figure()
plt.imshow(annotations, interpolation="nearest")
plt.title("Annotations")

# plt.figure()
# plt.imshow(
#     prediction, interpolation="nearest"
# )  # interpolation param passed for imshow not to apply it #vmin=1,vmax=3)
# plt.title("Prediction")

# overlay prediction on input image as grayscale with transparency
plt.figure()
plt.imshow(
    img_right.mean(axis=2), cmap="gray"
)  # if img_right has 3 channels, cmap is ignored by matplotlib
plt.imshow(prediction, alpha=0.5, cmap="viridis", interpolation="nearest")
plt.title("Prediction")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot DINOV2 features
print(type(features))
print(features.shape)  
# 17500 embeddings of 384 dimensions represent every annotated pixel in img_right!
# we are classifying those pixels!
features_image = np.reshape(
    features, (annotations.shape[0], annotations.shape[1], features.shape[1])
)


# %%
# Get embeddings for all pixels in the input image

# To get embeddings for all pixels, we need to pass an annotation map that labels all pixels
# with annotations "not blank" ---> features.shape = (17500, 384) -- > 17500 annotated pixels
# with annotations "blank" ---> features.shape = (370500, 384)
annotations_blank = np.ones(annotations.shape)

# get features for all pixels
features, _ = conv_paint.get_features_current_layers(
    np.moveaxis(img_left, -1, 0),  # move channels to the front
    annotations_blank, 
    model=model, 
    param=param
)

# make features a numpy array and reshape it to the shape of the input image
features_image = np.reshape(
    features, 
    (img_left.shape[0], img_left.shape[1], features.shape[1])  # image rows, cols, embedding dimensions
)

# %%
# choose some of the 384 coordinates of the pixel embeddings to plot
n_plot_rows = 8
n_plot_cols = 8
random_features = np.random.choice(
    features.shape[-1], n_plot_cols * n_plot_rows, replace=False
)

fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(features_image[:, :, random_features[i]], cmap="viridis")
    ax.set_title(f"Feature {random_features[i]}")  # coordinate of the embedding vector for each pixel

plt.tight_layout()

# %%
