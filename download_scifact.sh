#!/bin/bash

SCIENT_CLAIM_DIR="data/scifact"
ARCHIVE_URL="https://s3-us-west-2.amazonaws.com/scifact/release/latest/data.tar.gz"
ARCHIVE_NAME="data.tar.gz"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting Robust SciFact Download ==="

# --- Setup Directory ---
echo "Creating directory: $SCIENT_CLAIM_DIR"
mkdir -p "$SCIENT_CLAIM_DIR" || { echo "ERROR: Failed to create $SCIENT_CLAIM_DIR"; exit 1; }
cd "$SCIENT_CLAIM_DIR" || { echo "ERROR: Could not navigate to $SCIENT_CLAIM_DIR"; exit 1; }

# --- Download Archive ---
echo "Downloading archive from S3: $ARCHIVE_URL"
wget -q --show-progress "$ARCHIVE_URL" -O "$ARCHIVE_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: wget failed to download $ARCHIVE_NAME. Check network connection."
    cd ../..
    exit 1
fi

# --- Extract Files ---
echo "Extracting files..."
tar -xvzf "$ARCHIVE_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract $ARCHIVE_NAME. The downloaded file may be corrupt."
    cd ../..
    exit 1
fi

# --- Cleanup ---
echo "Removing archive file."
rm "$ARCHIVE_NAME"


cd ../.. # Return to project root

if $DOWNLOAD_SUCCESS; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === SciFact download complete and verified! ==="
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === SciFact download FAILED verification. ==="
    exit 1
fi