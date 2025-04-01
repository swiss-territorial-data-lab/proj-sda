#!/bin/bash
# Inference workflow

echo 'Run batch process to perform inference over several SWISSIMAGE years'
mkdir -p config/batch_process

SECOND=0
canton=CANTON                     # provide canton name in lower case
dl_model=68
for year in YEAR1 YEAR2 YEAR3     # list of years to process (no comma: YEAR1 YEAR2 YEAR3...)  

do
    echo '-----------'
    echo Canton = $canton
    echo Year = $year
    sed 's/#YEAR#/$year/g' config/config_det.template.yaml > config/batch_process/config_det_${year}_${canton}.yaml
    sed -i "s/SWISSIMAGE_YEAR/$year/g" config/batch_process/config_det_${year}_${canton}.yaml
    sed -i "s/CANTON/$canton/g" config/batch_process/config_det_${year}_${canton}.yaml
    sed -i "s/MODEL/$dl_model/g" config/batch_process/config_det_${year}_${canton}.yaml
    echo ' '
    echo 'prepare_aoi.py'
    python scripts/prepare_aoi.py config/batch_process/config_det_${year}_${canton}.yaml
    echo ' '
    file=data/AoI/$canton/aoi_${year}_${canton}.gpkg
    if [ -e $file ]; then
        echo ' '
        echo 'prepare_data.py'
        python scripts/prepare_data.py config/batch_process/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'generate_tilesets.py'
        stdl-objdet generate_tilesets config/batch_process/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'make_detections.py'
        stdl-objdet make_detections config/batch_process/config_det_${year}_${canton}.yaml
        echo ' '
        echo 'merge_detections.py'
        python scripts/merge_detections.py config/batch_process/config_det_${year}_${canton}.yaml
        echo ' '
    else
        echo File $file does not exist. Skip the processing.
        echo ' '
    fi
done
echo 'compile_years.py'
python scripts/compile_years.py config/batch_process/config_det_${year}_${canton}.yaml
echo ' '
echo 'merge_multi_results.py'   # Not useful, but rename the results to allow the use the same template than for several model
python scripts/merge_multi_results.py config/batch_process/config_det_${year}_${canton}.yaml
echo ' '
echo 'merge_across_years.py'
python scripts/merge_across_years.py config/batch_process/config_det_${year}_${canton}.yaml
echo 'remove_artifacts.py'
python ./scripts/remove_artifacts.py config/batch_process/config_det_${year}_${canton}.yaml
echo 'filter_detections.py'
python ./scripts/filter_detections.py config/batch_process/config_det_${year}_${canton}.yaml

duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."