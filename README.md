# Automatic detection of agricultaral soils affected by anthropegenic activities

The aim of the project is to automatically detect anthropogenic activities that have affected agricultural soils in the past. Two main categories have been defined: "non-agricultural activities" and "land movements". The results will make it possible to identify potentially rehabilitable soils that can be used to establish a land crop rotation map. <br>

**Table of content**

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Installation](#installation)
- [Getting started](#getting-started)
    - [Files structure](#files-structure)
    - [Data](#data)
    - [Scripts](#scripts)
    - [Workflow instructions](#workflow-instructions)
- [Disclaimer](#disclaimer)


## Requirements

### Hardware

The project has been run on a 32 GiB RAM machine with a 16 GiB GPU (NVIDIA Tesla T4) compatible with [CUDA](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 


### Software

- Ubuntu 20.04
- Python version 3.8 
- PyTorch version 1.10
- CUDA version 11.3
- GDAL version 3.0.4
<!-- - `object-detector` version [2.1.0](https://github.com/swiss-territorial-data-lab/object-detector/releases/tag/v.2.1.0)  -->
- [`object-detector`](https://github.com/swiss-territorial-data-lab/object-detector): clone the repo in the same folder than `proj-sda` and switch to branch `ch/multi-year`. 

### Installation

Install GDAL:

```
sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
```

Python dependencies can be installed with `pip` or `conda` using the `requirements.txt` file (compiled from `requirements.in`) provided. We advise using a [Python virtual environment](https://docs.python.org/3/library/venv.html).

- Create a Python virtual environment
```
$ python3 -m venv <dir_path>/<name of the virtual environment>
$ source <dir_path>/<name of the virtual environment>/bin/activate
```

- Install dependencies
```
$ pip install -r requirements.txt
```

- `requirements.txt` has been obtained by compiling `requirements.in`. Recompiling the file might lead to libraries version changes:
```
$ pip-compile requirements.in
```

## Getting started

### Files structure

The folders/files of the project `proj-sda` (in combination with `object-detector`) are organised as follows. Path names can be customised by the user, and * indicates numbers which may vary:

<pre>.
├── config                                          # configurations files folder
│   ├── config_det.yaml                             # detection workflow
│   ├── config_trne.yaml                            # training and evaluation workflow
│   └── detectron2_config_dqry.yaml                 # detectron 2
├── data                                            # folder containing the input data
│   └── ground_truth                                # available on S3/proj-sda/data/ground_truth
├── functions
│   └── fct_misc.py                                
├── output                                          # outputs folders
├── sandbox
│   ├── clip.py                                     # script clipping detections to the AoI
│   ├── filter_detections.py                        # script filtering the detections according to threshold values
│   ├── get_dem.sh                                  # batch script downloading the DEM of Switzerland
│   ├── mosaic.py                                   # script doing image mosaic
│   ├── plots.py                                    # script plotting figures
│   ├── rgb_to_greyscale.sh                         # script converting RGB images to greyscale images
│   └── track_detections.py                         # script tracking the detections in multiple years dataset 
├── scripts
│   ├── prepare_data.py                             # script preparing data to be processed by the object-detector scripts
│   └── rgb_to_greyscale.py                         # script converting RGB images to greyscale images
├── .gitignore                                      
├── LICENSE
├── README.md                                      
├── requirements.in                                 # list of python libraries required for the project
└── requirements.txt                                # python dependencies compiled from requirements.in file
</pre>


## Data

Below, the description of input data used for this project. 

- images: [_SWISSIMAGE Journey_](https://www.swisstopo.admin.ch/en/maps-data-online/maps-geodata-online/journey-through-time-images.html) is an annual dataset of aerial images of Switzerland. Only RGB images are used, from 1999 to current. It includes [_SWISSIMAGE 10 cm_](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html), _SWISSIMAGE 25 cm_ and _SWISSIMAGE 50 cm_. The images are downloaded from the [geo.admin.ch](https://www.geo.admin.ch/fr) server using [XYZ](https://developers.planet.com/docs/planetschool/xyz-tiles-and-slippy-maps/) connector. 
- ground truth: labels vectorized by domain experts, available on S3/proj-sda/data/ground_truth.


## Scripts

The `proj-sda` repository contains scripts to prepare and post-process the data and results:

1. `prepare_data.py`: format labels and produce tiles to be processed in the OD 
2. `rgb_to_greyscale.py`: convert RGB images to  greyscale images

Object detection is performed with tools present in the [`object-detector`](https://github.com/swiss-territorial-data-lab/object-detector) git repository. 


 ## Workflow instructions

The workflow can be executed by running the following list of actions and commands. Adjust the paths and input values of the configuration files accordingly. The contents of the configuration files in angle brackets must be assigned. 

**Training and evaluation**: 

```
$ python scripts/prepare_data.py config/config_trne.yaml
$ python ../object-detector/scripts/generate_tilesets.py config/config_trne.yaml
$ python ../object-detector/scripts/train_model.py config/config_trne.yaml
$ tensorboard --logdir output/output_trne/logs
```

Open the following link with a web browser: `http://localhost:6006` and identify the iteration minimising the validation loss and select the model accordingly (`model_*.pth`) in `config_trne`. For the provided parameters, `model_0002999.pth` is the default one.

```
$ python ../object-detector/scripts/make_detections.py config/config_trne.yaml
$ python ../object-detector/scripts/assess_detections.py config/config_trne.yaml
```

## Disclaimer

Depending on the end purpose, we strongly recommend users not take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all approaches based on deep learning.