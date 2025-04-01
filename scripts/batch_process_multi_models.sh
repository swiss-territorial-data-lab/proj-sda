#!/bin/bash
# Inference workflow

echo 'Run batch process to perfrom inference over several SWISSIMAGE years with 5 different models'

canton=CANTON                     # provide canton name in lowercase
for year in YEAR1 YEAR2 YEAR3     # list of years to process (no comma: YEAR1 YEAR2 YEAR3...)

do
    echo '-----------'
    echo Canton = $canton
    echo Year = $year
    sed 's/#YEAR#/$year/g' config/config_det.template.yaml > config/batch_process/config_det_${year}_${canton}.yaml
    sed -i "s/SWISSIMAGE_YEAR/$year/g" config/batch_process/config_det_${year}_${canton}.yaml
    sed -i "s/CANTON/$canton/g" config/batch_process/config_det_${year}_${canton}.yaml
    echo ' '
    echo 'prepare_aoi.py'
    python ./scripts/prepare_aoi.py config/batch_process/config_det_${year}_${canton}.yaml
    echo ' '
    file=data/AoI/$canton/aoi_${year}_${canton}.gpkg
    if [ -e $file ]; then
        echo ' '
        echo 'prepare_data.py'
        python ./scripts/prepare_data.py config/batch_process/config_det_${year}_${canton}.yaml
        tile_file=output/det/${canton}/${year}/tiles.geojson
        if [ -e $tile_file ]; then
            echo ' '
            echo 'generate_tilesets.py'
            stdl-objdet generate_tilesets config/batch_process/config_det_${year}_${canton}.yaml
            echo ' '
            for dl_model in 68 72 75 92 102
            do
                echo Model = $dl_model
                sed 's/#MODEL#/$dl_model/g' config/batch_process/config_det_${year}_${canton}.yaml > config/batch_process/config_det_${year}_${canton}_${dl_model}.yaml
                sed -i "s/MODEL/$dl_model/g" config/batch_process/config_det_${year}_${canton}_${dl_model}.yaml
                echo 'make_detections.py'
                stdl-objdet make_detections config/batch_process/config_det_${year}_${canton}_${dl_model}.yaml
                echo ' '
                echo 'merge_detections.py'
                python ./scripts/merge_detections.py config/batch_process/config_det_${year}_${canton}_${dl_model}.yaml
                echo ' '
            done
        fi
    else
        echo File $file does not exist. Skip the processing.
        echo ' '
    fi
done

for dl_model in 68 72 75 92 102

do
    echo 'compile_years.py'
    python ./scripts/compile_years.py config/batch_process/config_det_${year}_${canton}_${dl_model}.yaml
done

echo 'merge_multi_results.py'
python scripts/merge_multi_results.py config/batch_process/config_det_${year}_${canton}.yaml
echo ' '
echo 'merge_across_years.py'
python scripts/merge_across_years.py config/batch_process/config_det_${year}_${canton}.yaml
echo 'filter_detections.py'
python ./scripts/filter_detections.py config/batch_process/config_det_${year}_${canton}.yaml

duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."