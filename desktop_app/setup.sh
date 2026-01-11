#!/bin/bash
# Setup script for Financial Agent Desktop App
# This script sets up the Electron environment

echo "Setting up Financial Agent Desktop App..."

# Install dependencies in desktop_app
echo "Installing desktop app dependencies..."
npm install

# Install dependencies in frontend if not already installed
echo "Installing frontend dependencies..."
cd ../frontend
if [ ! -d "node_modules" ]; then
  npm install
fi

# Build frontend for production
echo "Building frontend..."
npm run build

# Go back to desktop_app
cd ../desktop_app

echo "Setup complete!"
echo ""
echo "To start development:"
echo "  npm run dev"
echo ""
echo "To build for distribution:"
echo "  npm run dist"
