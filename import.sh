#!/bin/bash
# filepath: process_markdown_images.sh

# Check if markdown file is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <markdown_file>"
  exit 1
fi

MD_FILE="$1"
if [ ! -f "$MD_FILE" ]; then
  echo "Error: File $MD_FILE does not exist"
  exit 1
fi

# Extract filename and date from the filename (expected format: YYYY-MM-DD-title.md)
FILENAME=$(basename "$MD_FILE" .md)
if [[ $FILENAME =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})-(.*) ]]; then
  POST_DATE="${BASH_REMATCH[1]} 00:00:00"
fi

# Create media directory if it doesn't exist
MEDIA_DIR="media/$FILENAME"
mkdir -p "$MEDIA_DIR"

# Default front matter properties
DEFAULT_FRONT_MATTER=$(
  cat <<EOF
---
title:
description:
author: ounols
date: '$POST_DATE'
categories: []
tags: []
pin: false
math: false
mermaid: false
image:
  path:
---
EOF
)

# Temporary files
TEMP_FILE=$(mktemp)
FRONT_MATTER_FILE=$(mktemp)

# Extract existing front matter
IN_FRONT_MATTER=false
HAS_FRONT_MATTER=false
while IFS= read -r line; do
  if [[ $line == "---" ]]; then
    if [ "$IN_FRONT_MATTER" = false ]; then
      IN_FRONT_MATTER=true
      HAS_FRONT_MATTER=true
      echo "$line" >>"$FRONT_MATTER_FILE"
      continue
    else
      IN_FRONT_MATTER=false
      echo "$line" >>"$FRONT_MATTER_FILE"
      continue
    fi
  fi

  if [ "$IN_FRONT_MATTER" = true ]; then
    echo "$line" >>"$FRONT_MATTER_FILE"
  fi
done <"$MD_FILE"

# If no front matter exists, add default one
if [ "$HAS_FRONT_MATTER" = false ]; then
  echo "$DEFAULT_FRONT_MATTER" >"$TEMP_FILE"
  cat "$MD_FILE" >>"$TEMP_FILE"
else
  # Check and add missing properties
  {
    echo "---"

    # Read existing front matter and check for required fields
    HAS_TITLE=false
    HAS_DESC=false
    HAS_AUTHOR=false
    HAS_DATE=false
    HAS_CATEGORIES=false
    HAS_TAGS=false
    HAS_PIN=false
    HAS_MATH=false
    HAS_MERMAID=false
    HAS_IMAGE=false

    while IFS= read -r line; do
      if [[ $line == "---" ]]; then continue; fi

      if [[ $line =~ ^title: ]]; then HAS_TITLE=true; fi
      if [[ $line =~ ^description: ]]; then HAS_DESC=true; fi
      if [[ $line =~ ^author: ]]; then HAS_AUTHOR=true; fi
      if [[ $line =~ ^date: ]]; then HAS_DATE=true; fi
      if [[ $line =~ ^categories: ]]; then HAS_CATEGORIES=true; fi
      if [[ $line =~ ^tags: ]]; then HAS_TAGS=true; fi
      if [[ $line =~ ^pin: ]]; then HAS_PIN=true; fi
      if [[ $line =~ ^math: ]]; then HAS_MATH=true; fi
      if [[ $line =~ ^mermaid: ]]; then HAS_MERMAID=true; fi
      if [[ $line =~ ^image: ]]; then HAS_IMAGE=true; fi

      if [[ $line != "---" ]]; then
        echo "$line"
      fi
    done <"$FRONT_MATTER_FILE"

    # Add missing properties
    [ "$HAS_TITLE" = false ] && echo "title: "
    [ "$HAS_DESC" = false ] && echo "description: "
    [ "$HAS_AUTHOR" = false ] && echo "author: ounols"
    [ "$HAS_DATE" = false ] && echo "date: '$POST_DATE'"
    [ "$HAS_CATEGORIES" = false ] && echo "categories: []"
    [ "$HAS_TAGS" = false ] && echo "tags: []"
    [ "$HAS_PIN" = false ] && echo "pin: false"
    [ "$HAS_MATH" = false ] && echo "math: false"
    [ "$HAS_MERMAID" = false ] && echo "mermaid: false"
    if [ "$HAS_IMAGE" = false ]; then
      echo "image:"
      echo "  path: "
    fi

    echo "---"

    # Add rest of the content
    IN_CONTENT=false
    while IFS= read -r line; do
      if [ "$IN_CONTENT" = true ]; then
        echo "$line"
      fi
      if [[ $line == "---" ]]; then
        IN_CONTENT=true
      fi
    done <"$MD_FILE"
  } >"$TEMP_FILE"
fi

# Process images
FINAL_TEMP=$(mktemp)
while IFS= read -r line; do
  if [[ $line =~ !\[.*\]\((.*)\) ]] || [[ $line =~ image:\s*path:\s*(.*) ]]; then
    if [[ $line =~ !\[.*\]\((.*)\) ]]; then
      IMG_URL="${BASH_REMATCH[1]}"
    else
      IMG_URL="${BASH_REMATCH[1]}"
    fi

    IMG_URL=${IMG_URL%\?*}
    IMG_URL=${IMG_URL%\}*}
    IMG_URL=${IMG_URL%\)*}

    if [[ $IMG_URL == /media/$FILENAME/* ]]; then
      echo "$line" >>"$FINAL_TEMP"
      continue
    fi

    IMG_FILENAME=$(basename "$IMG_URL")

    if [[ $IMG_URL =~ ^https?:// ]]; then
      curl -s "$IMG_URL" -o "$MEDIA_DIR/$IMG_FILENAME"
    else
      IMG_URL=${IMG_URL#/}
      if [ -f "$IMG_URL" ]; then
        cp "$IMG_URL" "$MEDIA_DIR/$IMG_FILENAME"
      fi
    fi

    NEW_PATH="/media/$FILENAME/$IMG_FILENAME"
    if [[ $line =~ !\[.*\]\((.*)\) ]]; then
      line=${line//$IMG_URL/$NEW_PATH}
    else
      line=${line//$IMG_URL/$NEW_PATH}
    fi
  fi
  echo "$line" >>"$FINAL_TEMP"
done <"$TEMP_FILE"

# Replace original file with processed content
mv "$FINAL_TEMP" "$MD_FILE"

# Cleanup temporary files
rm -f "$TEMP_FILE" "$FRONT_MATTER_FILE"

echo "Processing complete. Images have been saved to $MEDIA_DIR and front matter has been updated."
