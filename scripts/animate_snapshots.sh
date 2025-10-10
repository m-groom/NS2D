#!/bin/bash
# Animate write_*.png images using ffmpeg
module load ffmpeg

# Directory containing the PNG files
SNAPSHOT_DIR="../figures/snapshots/snapshots_s1/"

# Output video file
OUTPUT_FILE="snapshots.mp4"

# Frame rate (frames per second)
FPS=10

# Check if directory exists
if [ ! -d "$SNAPSHOT_DIR" ]; then
    echo "Error: Directory $SNAPSHOT_DIR does not exist"
    exit 1
fi

# Check if PNG files exist
if ! ls "$SNAPSHOT_DIR"/snapshot_*.png 1> /dev/null 2>&1; then
    echo "Error: No snapshot_*.png files found in $SNAPSHOT_DIR"
    exit 1
fi

# Create animation using ffmpeg
echo "Creating animation from PNG files in $SNAPSHOT_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Frame rate: $FPS fps"

ffmpeg -y \
    -framerate $FPS \
    -pattern_type glob \
    -i "$SNAPSHOT_DIR/snapshot_*.png" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -crf 18 \
    "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Animation created successfully: $OUTPUT_FILE"
else
    echo "Error: ffmpeg failed to create animation"
    exit 1
fi
