from rasterio.plot import show
from rasterio.merge import merge
import rasterio as rio
from pathlib import Path

path = Path('/mnt/data-03/cherny/git/proj-sda/output/oth-images/')
output_path = '/mnt/data-03/cherny/git/proj-sda/output/oth-images/mosaic_output.tif'

raster_files = list(path.iterdir())
raster_to_mosaic = []

for p in raster_files:
    raster = rio.open(p)
    raster_to_mosaic.append(raster)

mosaic, output = merge(raster_to_mosaic)

output_meta = raster.meta.copy()
output_meta.update(
    {"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    })

with rio.open(output_path, 'w', **output_meta) as m:
    m.write(mosaic)