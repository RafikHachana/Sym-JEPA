#!/bin/bash

# ATEPP Dataset Download and Extract Script
# Downloads Version 1.2 of the ATEPP dataset from Google Drive

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_URL="https://drive.google.com/file/d/1Df2KUdqvXtgvhvzx2D10YaQRbpP23rOS/view?usp=sharing"
FILE_ID="1Df2KUdqvXtgvhvzx2D10YaQRbpP23rOS"
OUTPUT_DIR="dataset"
ARCHIVE_NAME="ATEPP-dataset-v1.2.zip"

echo -e "${BLUE}ATEPP Dataset Download Script${NC}"
echo -e "${BLUE}===============================${NC}"
echo ""

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}gdown is not installed. Installing it now...${NC}"
    pip install gdown
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install gdown. Please install it manually:${NC}"
        echo -e "${RED}pip install gdown${NC}"
        exit 1
    fi
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${BLUE}Creating output directory: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
fi

# Change to output directory
cd "$OUTPUT_DIR"

# Download the dataset
echo -e "${BLUE}Downloading ATEPP dataset (Version 1.2)...${NC}"
echo -e "${YELLOW}This may take a while depending on your internet connection.${NC}"
echo ""

gdown "$FILE_ID" -O "$ARCHIVE_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to download the dataset. Please check your internet connection and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}Download completed successfully!${NC}"
echo ""

# Check if the downloaded file exists and has content
if [ ! -f "$ARCHIVE_NAME" ] || [ ! -s "$ARCHIVE_NAME" ]; then
    echo -e "${RED}Downloaded file is missing or empty. Please try downloading again.${NC}"
    exit 1
fi

# Determine file type and extract accordingly
echo -e "${BLUE}Determining file type...${NC}"
file_type=$(file "$ARCHIVE_NAME")

if [[ $file_type == *"Zip archive"* ]]; then
    echo -e "${BLUE}Extracting ZIP archive...${NC}"
    if command -v unzip &> /dev/null; then
        unzip -q "$ARCHIVE_NAME"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Extraction completed successfully!${NC}"
        else
            echo -e "${RED}Failed to extract ZIP archive.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}unzip command not found. Please install unzip to extract the archive.${NC}"
        exit 1
    fi
elif [[ $file_type == *"gzip compressed"* ]] || [[ $file_type == *"tar archive"* ]]; then
    echo -e "${BLUE}Extracting TAR/GZIP archive...${NC}"
    tar -xzf "$ARCHIVE_NAME"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Extraction completed successfully!${NC}"
    else
        echo -e "${RED}Failed to extract TAR/GZIP archive.${NC}"
        exit 1
    fi
elif [[ $file_type == *"RAR archive"* ]]; then
    echo -e "${BLUE}Extracting RAR archive...${NC}"
    if command -v unrar &> /dev/null; then
        unrar x "$ARCHIVE_NAME"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Extraction completed successfully!${NC}"
        else
            echo -e "${RED}Failed to extract RAR archive.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}unrar command not found. Please install unrar to extract the archive.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Unknown file type: $file_type${NC}"
    echo -e "${YELLOW}The file has been downloaded but not extracted. You may need to extract it manually.${NC}"
fi

# Show directory contents
echo ""
echo -e "${BLUE}Contents of the dataset directory:${NC}"
ls -la

# Optional: Remove the archive file to save space
echo ""
read -p "Do you want to remove the downloaded archive file to save space? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$ARCHIVE_NAME"
    echo -e "${GREEN}Archive file removed.${NC}"
fi

echo ""
echo -e "${GREEN}Dataset download and extraction completed!${NC}"
echo -e "${BLUE}Dataset location: $(pwd)${NC}"
echo ""
echo -e "${YELLOW}Please refer to the disclaimer.md file for usage terms and conditions.${NC}" 
