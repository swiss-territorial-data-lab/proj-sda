 #  Proj quarry detection and time machine
################################################################
#  Script used to run automatically Detections workflow for quarries detection
#  Inputs are defined in config_det.template.yaml


echo 'Run batch process to perfrom inference over several SWISSIMAGE years'

for year in 1971 1995 2004 2020      # list of years to process (no comma: YEAR1 YEAR2 YEAR3 ...)  
do
    echo ' '
    echo '-----------'
    echo Year = $year
    sed 's/#YEAR#/$year/g' config/config_det.template.yaml > config/config_det_$year.yaml
    sed -i "s/SWISSIMAGE_YEAR/$year/g" config/config_det_$year.yaml
    echo ' '
    echo 'prepare_data.py'
    python ./scripts/prepare_data.py config/config_det_$year.yaml
    echo ' '
    echo 'generate_tilesets.py'
    python ../object-detector/scripts/generate_tilesets.py config/config_det_$year.yaml
    # stdl-objdet generate_tilesets config/config_det_$year.yaml
    echo ' '
    echo 'make_detections.py'
    python ../object-detector/scripts/make_detections.py config/config_det_$year.yaml
    # stdl-objdet make_detections config/config_det_$year.yaml
    echo ' '
    echo 'merge_detections.py'
    python ./scripts/merge_detections.py config/config_det_$year.yaml
    echo ' '
    echo 'filter_detections.py'
    python ./scripts/filter_detections.py config/config_det_$year.yaml 
done