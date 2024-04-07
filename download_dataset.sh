#!/bin/bash

# Create the data directory
DATA_DIR=data
mkdir -p $DATA_DIR

# Download the dataset
wget http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz -O $DATA_DIR/shaded_images.tar.gz

# Extract the dataset
tar -xvzf $DATA_DIR/shaded_images.tar.gz -C $DATA_DIR