# Skel3D

<p align="center">
  [<a href="https://arxiv.org/abs/2412.03407"><strong>arXiv</strong></a>]
  [<a href="https://github.com/skel3d/skel3d"><strong>Project</strong></a>]
  [<a href="#citation"><strong>BibTeX</strong></a>]
</p>

## Description

This repository implements the training and testing tools for **Skel3D**. Given a single-view image and skeleton guidance, the proposed **Skel3D** synthesizes correct novel views without the need of an explicit 3D representation.

## Usage

### Installation

```bash
# create the container
apptainer build skel3d.sif build.def

# run container
apptainer exec --nv skel3d.sif bash
```

### Datasets

-   [Objaverse](https://objaverse.allenai.org/): For training / evaluating on Objaverse (7,729 instances for testing), please download the rendered dataset from [zero-1-to-3](https://github.com/cvlab-columbia/zero123). The original command they provided is:
    ```
    wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
    ```
    Unzip the data file and change `root_dir` in `configs/objaverse.yaml`.
-   [OmniObject3D](https://omniobject3d.github.io/): For evaluating on OmniObject3d (5,275 instances), please refer to [OmniObject3D Github](https://github.com/omniobject3d/OmniObject3D/tree/main), and change `root_dir` in `configs/omniobject3d`. Since we do not train the model on this dataset, we directly evaluate on the training set.
-   [GSO](https://app.gazebosim.org/miki/fuel/collections/Scanned%20Objects%20by%20Google%20Research): For evaluating on Google Scanned Objects (GSO, 1,030 instances), please download the whole 3D models, and use the rendered code from [zero-1-to-3](https://github.com/cvlab-columbia/zero123) to get 25 views for each scene. Then, change `root_dir` in `configs/googlescan.yaml` to the corresponding location. Our rendered files are available on [Google Drive](https://drive.google.com/file/d/1tV-qpiD5e-GzrjW5dQpTRviZa4YV326b/view?usp=drive_link).

### Inference

TBA

### Training

TBA

### Pretrained models

TBA

## Citation

If you find our code helpful, please cite our paper:

```
@article{fothar2024skel3d,
      author    = {Áron Fóthi, Bence Fazekas, Natabara Máté Gyöngyössy, Kristian Fenech},
      title     = {Skel3D: Skeleton Guided Novel View Synthesis},
      journal   = {arXiv},
      year      = {2024},
}
```
