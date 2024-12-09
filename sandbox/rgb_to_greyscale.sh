#!/bin/bash
# !!! Might not work yet in the provided venv

mypathin=<DIR_PATH>
mypathout=<DIR_PATH>
mkdir -p $mypathout

cd $mypathin
dir_list=$(ls -d -1 $PWD/*.tif) #create list of files

for i in $dir_list
do
    x=$(basename $i)
    # gdal_calc.py -R 16_33871_23265_RGB.tif --R_band=1 -G 16_33871_23265_RGB.tif --G_band=2 -B 16_33871_23265_RGB.tif --B_band=3 --outfile=16_33871_23265_gray_gdal_calc.tif --calc="R*0.2989+G*0.5870+B*0.1140"rm ./data/DEM/switzerland_dem.tif
    gdal_calc.py -R $mypathin/$x --R_band=1 -G $mypathin/$x --G_band=2 -B $mypathin/$x --B_band=3 --outfile=$mypathout/$x --calc="R*0.2989+G*0.5870+B*0.1140"
done