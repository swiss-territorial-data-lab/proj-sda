#!/bin/bash
# Download the SWISSIMAGE footprints

mkdir -p data/AoI/swissimage_footprints
wget https://data.geo.admin.ch/ch.swisstopo.swissimage-product.metadata/data.zip -P data/AoI/swissimage_footprints
unzip data/AoI/swissimage_footprints/data.zip -d data/AoI/swissimage_footprints
rm data/AoI/swissimage_footprints/data.zip