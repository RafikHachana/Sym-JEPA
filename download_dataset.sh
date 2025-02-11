#!/bin/bash
# download_dataset.sh
#
# This script downloads a dataset from a given URL and unzips it.
#
# Usage: ./download_dataset.sh <DATASET_URL> [destination_directory]
#
# If the destination directory is not provided, it defaults to "dataset".

set -e
set -o pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <DATASET_URL> [destination_directory]"
    exit 1
fi

DATASET_URL=$1
DEST_DIR=${2:-dataset}

echo "Creating destination directory '$DEST_DIR' if it does not exist..."
mkdir -p "$DEST_DIR"

FILE_NAME=$(basename "$DATASET_URL")
DOWNLOAD_PATH="$DEST_DIR/$FILE_NAME"

echo "Downloading dataset from '$DATASET_URL' to '$DOWNLOAD_PATH'..."
wget -O "$DOWNLOAD_PATH" "$DATASET_URL"

# Check if the downloaded file is a zip archive and unzip if it is.
if file "$DOWNLOAD_PATH" | grep -qi "Zip archive data"; then
    echo "Unzipping dataset..."
    unzip -o "$DOWNLOAD_PATH" -d "$DEST_DIR"
else
    echo "Downloaded file is not a zip archive. Skipping unzip."
fi

echo "Dataset has been downloaded and processed in '$DEST_DIR'." 