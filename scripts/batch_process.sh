#!/bin/bash
# Inference workflow

echo 'Run batch process to perfrom inference over several SWISSIMAGE years'

canton=CANTON                     # provide canton name
for year in YEAR1 YEAR2 YEAR3     # list of years to process (no comma: YEAR1 YEAR2 YEAR3...)  
do
    echo '-----------'
    echo Canton = $canton
    echo Year = $year
    sed 's/#YEAR#/$year/g' config/config_det.template.yaml > config/config_det_${year}_${canton}.yaml
    sed -i "s/SWISSIMAGE_YEAR/$year/g" config/config_det_${year}_${canton}.yaml
    sed -i "s/CANTON/$canton/g" config/config_det_${year}_${canton}.yaml
    echo ' '
    echo 'prepare_aoi.py'
    python ./scripts/prepare_aoi.py config/config_det_${year}_${canton}.yaml
    echo ' '
    file=./data/AoI/$canton/aoi_${year}_${canton}.gpkg
    if [ -e $file ]; then
        echo ' '
        echo 'prepare_data.py'
        python ./scripts/prepare_data.py config/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'generate_tilesets.py'
        stdl-objdet generate_tilesets config/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'make_detections.py'
        stdl-objdet make_detections config/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'merge_detections.py'
        python ./scripts/merge_detections.py config/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'filter_detections.py'
        python ./scripts/filter_detections.py config/config_det_${year}_${canton}.yaml 
    else
        echo File $file does not exist. Skip the processing.
        echo ' '
    fi
done