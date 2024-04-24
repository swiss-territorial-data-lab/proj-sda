#%%
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt
import numpy as np
img = rasterio.open('input/images/tests/zoom_15/15_2022_mosaic.tif')
# show(img)

full_img = img.read()

num_bands = img.count
print('Number of bands in the image = ', num_bands)

img_band1 = img.read(1)
img_band2 = img.read(2)
img_band3 = img.read(3)
#%%
# fig = plt.figure(figsize=(10,10))
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(img_band1, cmap='pink')
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(img_band2, cmap='pink')
# ax3 = fig.add_subplot(2,2,3)
# ax3.imshow(img_band3, cmap='pink') 
# plt.show()
print(('Coordinate reference system:', img.crs))

metadata = img.meta
print('Raster description:{desc}\n'.format(desc=metadata))
      
desc = img.descriptions
print('Raster description:{desc}\n'.format(desc=desc))

print('Geotransform:', img.transform)

# rasterio.plot.show_hist(full_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False)

# clipped_img = full_img[:, 300:900, 300:900]
clipped_img = full_img[:, :, :]
# plt.imshow(clipped_img[0,:,:]) 
# rasterio.plot.show_hist(clipped_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False)
# plt.show()

red_clipped = clipped_img[0].astype('f4') 
nir_clipped = clipped_img[1].astype('f4') 
ndvi_clipped = (nir_clipped - red_clipped) / (nir_clipped + red_clipped)

ndvi_clipped2 = np.divide(np.subtract(nir_clipped, red_clipped), np.add(nir_clipped, red_clipped))
ndvi_clipped3 = np.nan_to_num(ndvi_clipped2, nan=-1)
plt.imshow(img_band1, cmap='viridis')
plt.colorbar()
# plt.show()

# %%
