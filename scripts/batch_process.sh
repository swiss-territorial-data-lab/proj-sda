#  Proj quarry detection and time machine
################################################################
#  Script used to run automatically Detections workflow for quarries detection
#  Inputs are defined in config_det.template.yaml


echo 'Run batch processes to make quarries detection over several years'

for year in YEAR1 YEAR2 YEAR3       # list of years to process (no comma: YEAR1 YEAR2 YEAR3 ...) 
do
    echo ' '
    echo '-----------'
    echo Year = $year
    sed 's/#YEAR#/$year/g' config/config_det.template.yaml > config/config_det_$year.yaml
    sed -i "s/SWISSIMAGE_YEAR/$year/g" config/config_det_$year.yaml
    echo ' '
    echo 'prepare_data.py'
    python3 ./scripts/prepare_data.py config/config_det_$year.yaml
    echo ' '
    echo 'generate_tilesets.py'
    stdl-objdet generate_tilesets config/config_det_$year.yaml
    echo ' '
    echo 'make_detections.py'
    stdl-objdet make_detections config/config_det_$year.yaml
    echo ' '
    echo 'merge_detections.py'
    python3 ./scripts/merge_detections.py config/config_det_$year.yaml
    echo ' '
    echo 'filter_results.py'
    python3 ./scripts/filter_results.py config/config_det_$year.yaml
done