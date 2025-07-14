<a href=https://arxiv.org/abs/2412.15058><img src="https://img.shields.io/badge/arxiv-2312.07381-orange?logo=arxiv&logoColor=white"/></a>

# MultiverSeg

### [Project Page](https://multiverseg.csail.mit.edu) | [Paper](https://arxiv.org/abs/2412.15058) 

Official implementation of [MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](https://arxiv.org/abs/2412.15058) accepted at ICCV 2025

[Hallee E. Wong](https://halleewong.github.io/), [Jose Javier Gonzalez Ortiz](https://josejg.com/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://www.mit.edu/~adalca/)

![img](https://github.com/halleewong/MultiverSeg/blob/website/assets/teaser.png)

## Updates

* (2025-07-01) Checkout the 3DSlicer extension: https://github.com/dalcalab/SlicerMultiverSeg
* (2025-06-25) MultiverSeg was accepted to ICCV 2025!
* (2025-01-26) inference code and weights released
* (2024-12-19) preprint released!

## Models

We provide pre-trained weights [here](https://www.dropbox.com/scl/fo/71j9vl3d4db0u229rq689/AI_5oDICnt0HnBcry-xJSNQ?rlkey=7y42638h12ilqds8270owzric&st=3py413ys&dl=0).

## Installation

You can install `multiverseg` in two ways:

* **With pip**:

```
pip install git+https://github.com/halleewong/MultiverSeg.git
```

* **Manually**: cloning it and installing dependencies
```
git clone https://github.com/halleewong/MultiverSeg
python -m pip install -r ./MultiverSeg/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./MultiverSeg)"
```

## Getting Started

First download the model checkpoints 
```
cd checkpoints
./download.sh
```

Then see `./notebooks/inference.ipynb` for a tutorial. 

# Acknowledgements

This project builds extensively on code originally developed for [ScribblePrompt](https://github.com/halleewong/ScribblePrompt) and [UniverSeg](https://github.com/JJGO/UniverSeg)

# Citation

If you find our work or any of our materials useful, please cite our paper:
```
@article{wong2024multiverseg,
  title={MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance},
  author={Hallee E. Wong and Jose Javier Gonzalez Ortiz and John Guttag and Adrian V. Dalca},
  journal={arXiv preprint arXiv:2412.15058},
  year={2024},
}
```
