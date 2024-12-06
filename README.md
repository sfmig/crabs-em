# crabs-em
segmenting debris on crab EM data


Run the notebooks in the following environment:
```
conda create -n segment-env python=3.11 -y
pip install napari[all]
pip install torch torchvision torchaudio  # for Linux, pip, python CUDA 12.4 
pip install git+https://github.com/guiwitz/napari-convpaint.git@a30a35334dc38495444007e32f03393366792838
pip install ipympl  # for interactive plots
pip install imagecodecs  # to read full res tiff files
```

According to `napari-convpaint` installation [instructions](https://guiwitz.github.io/napari-convpaint/book/Installation.html#gpu), if using a GPU it is advisable to install the packages in the order above. For other platforms, see the Pytorch installation instructions [here](https://pytorch.org/get-started/locally/).

Since the `napari-convpaint` API is currently [changing](https://guiwitz.github.io/napari-convpaint/book/convpaint_api.html), I pinned it to the latest in `main` commit as of today.