# PyTorch code for MVCNN  
This is an updated version of the [pytorch implementation of MVCNN](https://github.com/jongchyisu/mvcnn_pytorch).

## Tested Versions
- Python 3.12
- PyTorch 2.2.1

## Get the dataset
    ./download_dataset.sh

## Install dependencies

First install poetry: https://python-poetry.org/docs/#installing-with-the-official-installer

Then use poetry to create a virtual environement and install dependencies.

    poetry install

## Configure training settings

Edit the file `train_mvcnn_settings.json`.

## Run training

    poetry run python train_mvcnn.py

## Edit prediction settings

Edit the file `predict_mvcnn_settings.json`.

## Run predictions

    poetry run python predict_mvcnn.py
  
## Additional Informations
[Project webpage](https://people.cs.umass.edu/~jcsu/papers/shape_recog/)  
[Depth Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/depth_images.tar.gz)  

[Blender script for rendering shaded images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)  
[Blender script for rendering depth images](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_depth.blend)  

## Reference
**A Deeper Look at 3D Shape Classifiers**  
Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji  
*Second Workshop on 3D Reconstruction Meets Semantics, ECCV, 2018*

**Multi-view Convolutional Neural Networks for 3D Shape Recognition**  
Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller,  
*International Conference on Computer Vision, ICCV, 2015*
