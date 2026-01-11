#!/bin/bash
# Icon generation helper for different platforms
# You'll need ImageMagick and icnsutils installed

# Generate from a single PNG (recommended 512x512 or larger)
SOURCE_ICON="icon.png"

# For Windows (ICO format)
# Using ImageMagick: convert icon.png -define icon:auto-resize=256,128,96,64,48,32,16 icon.ico

# For macOS (ICNS format)
# Using icnsutils: png2icns icon.icns icon.png

# For Linux (PNG is fine)
# Just use icon.png (256x256 or larger)

echo "To generate icons:"
echo "1. Prepare a high-resolution image (512x512 or larger)"
echo "2. Windows: convert icon.png -define icon:auto-resize=256,128,96,64,48,32,16 icon.ico"
echo "3. macOS: png2icns icon.icns icon.png"
echo "4. Linux: cp icon.png icon.png (256x256 minimum)"
