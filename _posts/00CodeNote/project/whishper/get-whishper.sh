#!/bin/bash
set -e

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if docker is installed
if ! [ -x "$(command -v docker)" ]; then
    echo -e "${RED}❌ Docker is not installed. Please install docker first.${NC}"
    echo -e "> Run official docker installation script? (y/n)"
    echo -e "---> Check out the script: https://get.docker.com"
    echo -e "---> I will run: 'curl -fsSL https://get.docker.com | sudo sh'"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        echo -e "${BLUE}ℹ️  Running official docker installation script${NC}"
        curl -fsSL https://get.docker.com | sudo sh
    else
        echo -e "${RED}❌ Aborting installation${NC}"
        exit 1
    fi
fi

# Ask if user wants to get everything in the current directory or in a new directory
echo -e "${BLUE}> Do you want to set up everything in the current directory? (y/n)${NC}"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo -e "ℹ️  Setting up everything in the current directory"
else
    echo -e "${BLUE}> Enter the name of the directory where you want to set up everything:${NC}"
    read directory
    echo -e "ℹ️  Setting up everything in the $directory directory"
    mkdir $directory
    cd $directory
fi

echo ""
echo -e "ℹ️  For more info about GPU support check out: https://whishper.net/guides/gpu/"
echo -e "${BLUE}>  Do you want to use the GPU version? (y/n)${NC}"
read gpu
if [ "$gpu" != "${gpu#[Yy]}" ] ;then
    gpu=true
    if ! [ -x "$(command -v nvidia-container-toolkit)" ]; then
        echo ""
        echo -e "${YELLOW}⚠️  WARNING: nvidia-container-toolkit seems to not be installed. You must install it for whishper to work with GPU. Read more at: https://whishper.net/guides/gpu/${NC}"
        echo -e "${BLUE}>  Do you want to continue anyways? (y/n)"
        read continue
        if [ "$gpu" != "${gpu#[Yy]}" ] ;then
            echo -e "ℹ️  Continuing..."
        else
            echo -e "${RED}❌ Aborting installation${NC}"
            exit 1
        fi
    fi
else
    gpu=false
fi

echo ""
echo -e "ℹ️  Getting the docker-compose.yml file from Github"
if [ "$gpu" = true ] ;then
    curl -o docker-compose.yml https://raw.githubusercontent.com/pluja/whishper/main/docker-compose.gpu.yml > /dev/null 2>&1
else
    curl -o docker-compose.yml https://raw.githubusercontent.com/pluja/whishper/main/docker-compose.yml > /dev/null 2>&1
fi
sleep 1

# check if .env exists
if [ -f .env ]; then
    echo ""
    echo -e "${YELLOW}⚠️  .env file already exists${NC}"
    echo -e "${BLUE}> Do you want to overwrite it? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        echo -e "ℹ️  Copying env.example to .env"
        cp .env .env.backup
        curl -o .env https://raw.githubusercontent.com/pluja/whishper/main/example.env > /dev/null 2>&1
        sleep 1
    fi
    echo ""
else
    echo -e "ℹ️  Getting the default .env file from Github"
    curl -o .env https://raw.githubusercontent.com/pluja/whishper/main/example.env > /dev/null 2>&1
    sleep 1
fi

# Create necessary directories for libretranslate
echo -e "ℹ️  Creating necessary directories for libretranslate"
sudo mkdir -p ./whishper_data/libretranslate/{data,cache}
sleep 1

# This permissions are for libretranslate docker container
echo -e "ℹ️  Setting permissions for libretranslate"
case "$OSTYPE" in
  darwin*)  echo -e "ℹ️  macOS detected... Leaving permissions untouched." ;;
  linux*)   sudo chown -R 1032:1032 ./whishper_data/libretranslate ;;
  *)        echo -e "${YELLOW}⚠️  unknown: $OSTYPE${NC}" ;;
esac
sleep 1

echo ""
echo -e "${BLUE}>  Do you want to pull the docker images? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo -e "ℹ️  Pulling and building docker images"
    sudo docker compose pull
fi

echo ""
echo -e "${BLUE}>  Do you want to start the containers now? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo -e "ℹ️  Starting whishper..."
    sudo docker compose up -d
fi
