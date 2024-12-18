# Automatic detection of agricultural soils affected by anthropogenic activities

The aim of the project is to automatically detect anthropogenic activities that have affected agricultural soils in the past. Two main categories have been defined: "non-agricultural activity" and "land movement". The results will make it possible to identify potentially rehabilitable soils that can be used to establish a land crop rotation map. <br>
This project was developed in collaboration with the Canton of Ticino and of the Canton of Vaud.

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
- object-detector version [2.3.2](https://github.com/swiss-territorial-data-lab/object-detector/releases/tag/v2.3.2)

### Installation

Install GDAL:

```
sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
```

Python dependencies can be installed with `pip` or `conda` using the `requirements.txt` file (compiled from `requirements.in`) provided. We advise using a [Python virtual environment](https://docs.python.org/3/library/venv.html).

- Create a Python virtual environment
```
$ python3.8 -m venv <dir_path>/<name of the virtual environment>
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

The folders/files of the project `proj-sda` (in combination with the `object-detector`) are organised as follows. Path names can be customised by the user, and * indicates numbers which may vary:

<pre>.
├── config                                          # configurations files folder
│   ├── config_det.template.yaml                    # detection workflow template
│   ├── config_det.yaml                             # detection workflow
│   ├── config_sandbox.yaml                         # sandbox workflow
│   ├── config_trne.yaml                            # training and evaluation workflow
│   └── detectron2_config_dqry.yaml                 # detectron 2
├── data                                            # folder containing the input data
│   ├── AoI                                         # available on request
│   ├── DEM
│   ├── empty_tiles                     
│   ├── FP
│   ├── ground_truth                                # available on request                              
│   ├── layers                                      # available on request 
│   └── categories_ids.json                         # class dictionnary     
├── functions
│   ├── constants.py                  
│   ├── fct_metrics.py                             
│   └── fct_misc.py                                
├── images                                          
├── models                                          # trained models
├── output                                          # outputs folders
│   ├── det                            
│   └── trne
├── sandbox
│   ├── clip.py                                     # script clipping detections to the AoI
│   ├── gt_analysis.py                              # script plotting GT characteristics
│   ├── match_colour.py                             # script matching colour histogram to a reference image
│   ├── mosaic.py                                   # script mosaicking images
│   ├── rgb_to_greyscale.py                         # script converting RGB images to greyscale images
│   ├── rgb_to_greyscale.sh                         # script converting RGB images to greyscale images
│   └── tiff2geotiff.py                             # convert tiff to geotiff
dataset 
├── scripts
│   ├── batch_process.sh                            # script to execute several commands
│   ├── filter_detections.py                        # script detections filtering 
│   ├── get_dem.sh                                  # script downloading swiss DEM and converting it to EPSG:2056
│   ├── merge_detections.py                         # script merging adjacent detections and attributing class
│   ├── merge_years.py                              # script merging all year detections layers
│   ├── prepare_aoi.py                              # script preparing the aoi shapefile for inference
│   ├── prepare_data.py                             # script preparing data to be processed by the object-detector scripts
│   └── result_analysis.py                          # script plotting some parameters
├── .gitignore                                      
├── LICENSE
├── README.md                                      
├── requirements.in                                 # list of python libraries required for the project
└── requirements.txt                                # python dependencies compiled from requirements.in file
</pre>


## Data

Below, the description of input data used for this project. 

- images: [_SWISSIMAGE Journey_](https://map.geo.admin.ch/#/map?lang=fr&center=2660000,1190000&z=1&bgLayer=ch.swisstopo.pixelkarte-farbe&topic=ech&layers=ch.swisstopo.swissimage-product@year=2024;ch.swisstopo.swissimage-product.metadata@year=2024) is an annual dataset of aerial images of Switzerland from 1946 to today. The images are downloaded from the [geo.admin.ch](https://www.geo.admin.ch/fr) server using [XYZ](https://api3.geo.admin.ch/services/sdiservices.html#xyz) connector. 
- swissimage footprints: image acquisition footprints by year (swissimage_footprint_*.shp) can be found [here](https://map.geo.admin.ch/#/map?lang=fr&center=2660000,1190000&z=1&bgLayer=ch.swisstopo.pixelkarte-farbe&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.astra.wanderland-sperrungen_umleitungen,f;ch.swisstopo.swissimage-product@year=2021;ch.swisstopo.swissimage-product.metadata@year=2021&timeSlider=2021). 
- canton: shapefile of the canton's borders used to define the AoI. The limits of Canton of Ticino and Canton of Vaud are available on request
- ground truth: labels vectorised by the domain experts. Available on request.
- layers: list of vector layers provided by the domain experts to be spatially intersect with the results to either excluded or to add intersection information in the final attribute table. Available on request.
- category_ids.json: categories attributed to the detections.
- models: the trained models used to produce the results presented in the documentation is available on request.

## Scripts

The `proj-sda` repository contains scripts to prepare and post-process the data and results. Hereafter a short description of each script and a workflow graph:

<p align="center">
<img src="./images/sda_workflow_graph.png?raw=true" width="100%">
<br />
</p>

1. `prepare_aoi.py`: produce an aoi shapefile compatible with a SWISSIMAGE year and desired geographical boundaries.
2. `prepare_data.py`: format labels and produce tiles to be processed in the OD.
3. `results_analysis.py`: plot some parameters of the detections to help understand the results (optional).
4. `merge_detections.py`: merge adjacent detections cut by tiles into a single detection and attribute the class (the class of the maximum area).
5. `filter_detections.py`: filter detections by overlap with other vector layers. The overlapping portion of the detection can be removed or a new attribute column is created to indicate the overlapping ratio with the layer of interest. Other information such as score, elevation, slope are also displayed.
6. `merge_years.py`: merge all the detection layers obtained during inference by year.
7. `get_dem.sh`: download the DEM of Switzerland.
8. `batch_process.sh`: batch script to perform the inference workflow over several years.

Object detection is performed with tools present in the [`object-detector`](https://github.com/swiss-territorial-data-lab/object-detector) git repository. 


 ## Workflow instructions

The workflow can be executed by running the following list of actions and commands. Adjust the paths and input values of the configuration files accordingly. The contents of the configuration files in square brackets must be assigned. 

**Training and evaluation**: 

Prepare the data:
```
$ python scripts/prepare_data.py config/config_trne.yaml
$ stdl-objdet generate_tilesets config/config_trne.yaml
```

Train the model:
```
$ stdl-objdet train_model config/config_trne.yaml
$ tensorboard --logdir output/trne/logs
```

Open the following link with a web browser: `http://localhost:6006` and identify the iteration minimising the validation loss and select the model accordingly (`model_*.pth`) in `config_trne`. For the provided parameters, `model_0004999.pth` is the default one.

Perform and assess detections:
```
$ stdl-objdet make_detections config/config_trne.yaml
$ stdl-objdet assess_detections config/config_trne.yaml
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

Colour processing on images can be performed if needed prior to inference. <br>
Copy the selected trained model to the folder `models`.
 
```
$ python scripts/prepare_aoi.py config/config_det.yaml
$ python scripts/prepare_data.py config/config_det.yaml
$ stdl-objdet generate_tilesets config/config_det.yaml
$ stdl-objdet make_detections config/config_det.yaml
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

## Sandbox

Additional scripts can be used to process images. Their use is optional.

1. `clip.py`: clip a vecor layer with another one.
2. `gt_analysis.py`: plot GT characteristics.
3. `match_colour.py`: normalise the colour histogram to the one of a reference image. 
4. `mosaic.py`: mosaic images. 
5. `rgb_to_greyscale.py`: convert RGB images to greyscale images. 
6. `rgb_to_greyscale.sh`: convert RGB images to greyscale images. 
7. `tiff2geotiff.py`: convert tiff to geotiff files. Provide geotiff files with the same size and extent as the tiff files to georeference. 

This project uses a multi-year dataset comprising greyscale and RGB images. The historical greyscale images are colourised using the method developed by [Farella et al. 2022](https://doi.org/10.3390/jimaging8100269), for which the code is available [here](https://github.com/3DOM-FBK/Hyper_U_Net).


## Disclaimer

Depending on the end purpose, we strongly recommend users not take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all approaches based on deep learning.
