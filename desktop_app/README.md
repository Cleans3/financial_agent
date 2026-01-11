# Financial Agent Desktop App

A cross-platform Electron desktop application for the Financial Agent UI.

## Features

- ✅ Cross-platform support (Windows, macOS, Linux)
- ✅ Clean UI without menu bar
- ✅ Single window with smooth page transitions
- ✅ Identical display to web frontend
- ✅ Automatic API proxying to backend
- ✅ Development and production modes

## Prerequisites

- Node.js 16+ and npm
- Python backend running on `http://localhost:8000`

## Installation

```bash
# Install dependencies (run from desktop_app folder)
npm install

# Also ensure frontend dependencies are installed
cd ../frontend
npm install
cd ../desktop_app
```

## Development

### Option 1: Run with automatic frontend dev server

```bash
npm run dev
```

This will:
- Start the frontend dev server on port 3000
- Wait for it to be ready
- Launch Electron connected to the dev server
- Open DevTools automatically

### Option 2: Manual setup

Terminal 1 (frontend):
```bash
cd ../frontend
npm run dev
```

Terminal 2 (Electron):
```bash
npm start
```

## Building

### Build for your current platform

```bash
npm run dist
```

### Build for specific platform

```bash
npm run dist-win    # Windows
npm run dist-mac    # macOS
npm run dist-linux  # Linux
```

Built files will be in the `dist` folder.

## Configuration

### Main Process

- **main.js**: Electron main process, window creation, menu configuration
- **preload.js**: Security-focused preload script for renderer process
- **package.json**: Build and dependency configuration

### Window Settings

Edit `main.js` to customize:
- Window size: `width`, `height`, `minWidth`, `minHeight`
- Icon paths in `assets/` folder
- DevTools behavior (remove `mainWindow.webContents.openDevTools()` to disable)

### API Proxy

The app proxies `/api` requests to `http://localhost:8000`. Edit the proxy target in frontend's `vite.config.js` if needed.

## Platform-Specific Notes

### Windows
- Creates both NSIS installer and portable .exe
- Requires admin rights for NSIS installation
- Squirrel.Windows support for auto-updates

### macOS
- Creates DMG installer and ZIP archive
- Code signing recommended for distribution
- App should be notarized for Monterey+

### Linux
- Creates AppImage and DEB package
- Icon should be at least 256x256 pixels

## Troubleshooting

### App won't start

1. Check that frontend dev server is running: `http://localhost:3000`
2. Verify backend API is accessible: `http://localhost:8000`
3. Check console in DevTools for errors

### Blank window

1. Ensure frontend built successfully
2. Check that `dist/index.html` exists
3. Verify paths in `main.js` are correct

### API requests fail

1. Ensure Python backend is running
2. Check `http://localhost:8000/api` is accessible
3. Verify CORS headers are correct

## Assets

Place application icons in the `assets/` folder:
- **Windows**: `icon.ico` (256x256+)
- **macOS**: `icon.icns`
- **Linux**: `icon.png` (256x256+)

## Project Structure

```
desktop_app/
├── main.js              # Electron main process
├── preload.js           # Security preload
├── package.json         # Dependencies and scripts
├── assets/              # Application icons
└── README.md           # This file
```

## Performance Tips

1. Disable DevTools in production (remove from main.js)
2. Use code signing for Windows to avoid SmartScreen warnings
3. Test on actual hardware before distribution
4. Consider lazy-loading large components

## License

Same as main Financial Agent project
