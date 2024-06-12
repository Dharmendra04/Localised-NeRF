# Localised-NeRF: Specular Highlights and Colour Gradient Localising in NeRF

This repository contains the public source code release for the paper [Specular Highlights and Colour Gradient Localising in NeRF(Localised-NeRF)](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FCVPR2024W%2FNRI%2Fpapers%2FSelvaratnam_Localised-NeRF_Specular_Highlights_and_Colour_Gradient_Localising_in_NeRF_CVPRW_2024_paper.pdf&data=05%7C02%7Cdharmendra.selvaratnam%40postgrad.plymouth.ac.uk%7Ca3f0afc6ebee4cf38fe108dc8af26b87%7C5437e7eb83fb4d1abfd3bb247e061bf1%7C1%7C0%7C638538020987991449%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=ue1zHZXcfCPxLGwDwNBqL7xtVtyolD1t1ptrHzWzjpU%3D&reserved=0). This project is based on
[JAXNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf),
which is a [JAX](https://github.com/google/jax) implementation of
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://www.matthewtancik.com/nerf). This code is created and maintained by [Dharmendra Selvaratnam](https://github.com/Dharmendra04?tab=repositories) & [Dena Bazazian](https://denabazazian.github.io/)


## Abstract

Neural Radiance Field (NeRF) based systems predominantly operate within the RGB (Red, Green, and Blue) space; however, the distinctive capability of the HSV (Hue, Saturation, and Value) space to discern between specular and diffuse regions is seldom utilised in the literature. We introduce Localised-NeRF, which projects the queried pixel point onto multiple training images to obtain a multi-view feature representation on HSV space and gradient space to obtain important features that can be used to synthesise novel view colour. This integration is pivotal in identifying specular highlights within scenes, thereby enriching the model's understanding of specular changes as the viewing angle alters. Our proposed Localised-NeRF model uses an attention-driven approach that not only maintains local view direction consistency but also leverages image-based features namely the HSV colour space and colour gradients. These features serve as effective indirect priors for both the training and testing phases to predict the diffuse and specular colour.



## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```

# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name localised_nerf python=3.11.7; conda activate localised_nerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r requirements.txt
```

[Optional] Install GPU and TPU support for Jax
```
pip install --upgrade pip

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Follow one of the above stated code accroding to your CUDA compatibiltiy
```


## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip`.

from the [Shiny Blender official Google Drive](from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip shiny_blender_dataset from the website).



## Running

To quickly try the pipeline out you can use the demo config (configs/demo),
however you will need the full configs (configs/blender or configs/llff) to
replicate our results.

The first step is to train a deferred NeRF network:

```
python -m localised_nerf.train \
  --data_dir=/PATH/TO/YOUR/SCENE/DATA \
  --train_dir=/PATH/TO/THE/PLACE/YOU/WANT/TO/SAVE/CHECKPOINTS \
  --config=configs/blender
```

keep in mind the above code should be implemented one folder ahead of where you save the localised_nerf folder. For example - if you save in Desktop, then you have to implement the code under Desktop directory, not in Desktop/localised_nerf directory.

## Running out of memory

If the batch size or chunk size  is high it will make storage problem. All our experiment results were done by using 64 batch size and chunk size. you can adjust based on your storage availability.

## Visualising results

you can visualise the results by using the follwing code on tensor-board.

```tensorboard --logdir ./ ``` [current directory is the training checkpoint directory ] or  using ```tensorboard --logdir=$EVENTS_FOLDER ``` [from any directory].

You can see the test results, features and Metrics etc.


# BibTeX

If you find our code useful, please cite our paper:
```
@InProceedings{Selvaratnam_2024_CVPR,
    author    = {Selvaratnam, Dharmendra and Bazazian, Dena},
    title     = {Localised-NeRF: Specular Highlights and Colour Gradient Localising in NeRF},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {2791--2801}
}
```


