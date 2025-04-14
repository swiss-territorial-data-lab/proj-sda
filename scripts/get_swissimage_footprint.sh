#!/bin/bash
# Download the SWISSIMAGE footprints

mkdir -p data/AoI/swissimage_footprint
wget https://data.geo.admin.ch/ch.swisstopo.swissimage-product.metadata/data.zip -P data/AoI/swissimage_footprint
unzip data/AoI/swissimage_footprint/data.zip -d data/AoI/swissimage_footprint
rm data/AoI/swissimage_footprint/data.zip