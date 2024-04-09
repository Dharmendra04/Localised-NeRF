# Localised-NeRF: Specular Highlights and Colour Gradient Localising in NeRF

This project is based on
[JAXNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf),
which is a [JAX](https://github.com/google/jax) implementation of
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://www.matthewtancik.com/nerf).


## Abstract

In contemporary NeRF approaches, the analysis of queried points occurs in isolation, disregarding the feature consistency across neighbouring rays. In this paper, we tackle this challenge to enhance it by incorporating additional insights from surrounding views. We introduce Localised-NeRF, which projects the queried pixel point onto multiple training images to create a multi-view feature representation and integrates a ray-based attention mechanism with the HSV colour space and colour gradients for determining the final colour output. This integration is pivotal in identifying specular highlights within scenes, thereby enriching the model's understanding of specular changes as the viewing angle alters. Our proposed Localised-NeRF model uses an attention-driven approach that not only maintains local view direction consistency but also leverages image-based features namely the HSV colour space and colour gradients. These features serve as effective indirect priors for both the training and testing phases. Moreover, an attention strategy among rays captures local information along the pixels, facilitating the accurate prediction of the final RGB colour. Our model exhibits competitive performance with prior attention-based multi-view feature extraction NeRF models, as demonstrated on the Shiny Blender and Synthetic datasets.

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


