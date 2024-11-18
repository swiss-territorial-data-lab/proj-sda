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
│   ├── config_det.template.yaml                    # detection workflow template
│   ├── config_det.yaml                             # detection workflow
│   ├── config_sandbox.yaml                         # sandbox workflow
│   ├── config_trne.yaml                            # training and evaluation workflow
│   └── detectron2_config_dqry.yaml                 # detectron 2
├── data                                            # folder containing the input data
│   └── ground_truth                                # available on S3/proj-sda/data/ground_truth
├── functions
│   ├── constants.py                  
│   ├── fct_metrics.py                             
│   └── fct_misc.py                                
├── output                                          # outputs folders
├── sandbox
│   ├── clip.py                                     # script clipping detections to the AoI
│   ├── gt_analysis.py                              # script plotting GT characteristics
│   ├── mosaic.py                                   # script mosaicking images
│   └── rgb_to_greyscale.sh                         # script converting RGB images to greyscale images
dataset 
├── scripts
│   ├── batch_process.sh                            # script to execute several commands
│   ├── filter_detections.py                        # script detections filtering 
│   ├── get_dem.sh                                  # script downloading swiss DEM and converting it to EPSG:2056
│   ├── match_colour.py                             # script matching colour histogram to a reference image
│   ├── merge_detections.py                         # script merging adjacent detections and attributing class
│   ├── merge_years.py                              # script merging all year detections layers
│   ├── prepare_data.py                             # script preparing data to be processed by the object-detector scripts
│   ├── result_analysis.py                          # script plotting some parameters
│   └── rgb_to_greyscale.py                         # script converting RGB images to greyscale images
├── .gitignore                                      
├── LICENSE
├── README.md                                      
├── requirements.in                                 # list of python libraries required for the project
└── requirements.txt                                # python dependencies compiled from requirements.in file
</pre>


## Data

Below, the description of input data used for this project. 

- images: [_SWISSIMAGE Journey_](https://www.swisstopo.admin.ch/en/maps-data-online/maps-geodata-online/journey-through-time-images.html) is an annual dataset of aerial images of Switzerland from 1946 to today. The images are downloaded from the [geo.admin.ch](https://www.geo.admin.ch/fr) server using [XYZ](https://developers.planet.com/docs/planetschool/xyz-tiles-and-slippy-maps/) connector. 
- ground truth: labels vectorized by the domain experts, available on S3/proj-sda/data/ground_truth/.
- layers: list of vector layers provided by the domain experts to be spatially intersect with the results to either excluded or to add intersection information in the final attribute table, available on S3/proj-sda/data/layers/.
- category_ids.json: categories attributed to the detections, available on S3/proj-sda/data/.

## Scripts

The `proj-sda` repository contains scripts to prepare and post-process the data and results. Hereafter a short description of each script and a workflow graph:

<p align="center">
<img src="./images/sda_workflow_graph.png?raw=true" width="100%">
<br />
</p>

1. `prepare_data.py`: format labels and produce tiles to be processed in the OD 
2. `rgb_to_greyscale.py`: convert RGB images to greyscale images (optional)
3. `match_colour.py`: normalise the colour histogram to the one of a reference image (optional). It can be used for instance after the colourisation of greyscale images to match the RGB images colours.
4. `results_analysis.py`: plot some parameters of the detections to help understand the results (optional)
5. `merge_detections.py`: merge adjacent detections cut by tiles into a single detection and attribute the class (the class of the maximum area)
6. `filter_detections.py`: filter detections by overlap with other vector layers. The overlapping portion of the detection can be removed or a new attribute column is created to indicate the overlapping ratio with the layer of interest. Other information such as score, elevation, slope are also displayed.
7. `merge_years.py`: merge all the detection layers obtained during inference by year.

Object detection is performed with tools present in the [`object-detector`](https://github.com/swiss-territorial-data-lab/object-detector) git repository. 


 ## Workflow instructions

The workflow can be executed by running the following list of actions and commands. Adjust the paths and input values of the configuration files accordingly. The contents of the configuration files in angle brackets must be assigned. 

**Training and evaluation**: 

Prepare the data:
```
$ python scripts/prepare_data.py config/config_trne.yaml
$ python ../object-detector/scripts/generate_tilesets.py config/config_trne.yaml
```

Optional: the images can be standardise by applying 
```
$ python scripts/rgb_to_greyscale.py config/config_trne.yaml
$ python scripts/match_colour.py config/config_trne.yaml
```

Train the model:
```
$ python ../object-detector/scripts/train_model.py config/config_trne.yaml
$ tensorboard --logdir output/trne/logs
```

Open the following link with a web browser: `http://localhost:6006` and identify the iteration minimising the validation loss and select the model accordingly (`model_*.pth`) in `config_trne`. For the provided parameters, `model_0001999.pth` is the default one.

Perform and assess detections:
```
$ python ../object-detector/scripts/make_detections.py config/config_trne.yaml
$ python ../object-detector/scripts/assess_detections.py config/config_trne.yaml
```

Some characteristics of the detections can be analysed with the help of plots:
```
$ python scripts/result_analysis.py config/config_trne.yaml
```

Finally, the detection obtained by tiles can be merged when adjacent and a new assessment is performed:
```
$ python scripts/merge_detections.py config/config_trne.yaml
```

**Inference**: 

Colour processing on images can be performed if needed prior to inference.
 
```
$ python scripts/prepare_data.py config/config_det.yaml
$ python ../object-detector/scripts/generate_tilesets.py config/config_det.yaml
$ python ../object-detector/scripts/make_detections.py config/config_det.yaml
$ python scripts/merge_detections.py config/config_det.yaml
$ scripts/get_dem.sh
$ python scripts/filter_detections.py config/config_det.yaml
```

The inference workflow has been automated and can be run for a batch of years (to be specified in the script) by executing these commands:
```
$ scripts/get_dem.sh
$ scripts/batch_process.sh
```

Finally, all the detection layers obtained for each year are merged into a single geopackage.

```
$ python scripts/merge_years.py config/config_det.yaml
```

## Disclaimer

Depending on the end purpose, we strongly recommend users not take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all approaches based on deep learning.