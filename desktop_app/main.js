const { app, BrowserWindow, Menu } = require('electron');
const isDev = require('electron-is-dev');
const path = require('path');

// Enable V8 code caching for faster startup
app.setPath('userData', path.join(app.getPath('appData'), 'FinancialAgent'));

// Enable GPU acceleration globally
app.disableHardwareAcceleration = false;

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 600,
    icon: path.join(__dirname, process.platform === 'win32' ? 'assets/icon.ico' : 'assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      enableRemoteModule: false,
      // Performance optimizations
      v8CodeCaching: true,
      nodeIntegrationInWorker: false,
      contextIsolationInWorker: true,
    },
    // Enable V-Sync for smooth 60 FPS
    vsync: true,
  });

  // Enable hardware acceleration
  app.commandLine.appendSwitch('enable-hardware-acceleration');
  
  // Disable menu bar for a neat look
  Menu.setApplicationMenu(null);

  // Load the appropriate URL based on environment
  const startUrl = isDev
    ? 'http://localhost:3000' // Dev server
    : `file://${path.join(__dirname, '../frontend/dist/index.html')}`; // Built app

  console.log('Loading URL:', startUrl);
  mainWindow.loadURL(startUrl);
  
  // Handle loading errors - fallback to dev server if file protocol fails
  mainWindow.webContents.on('did-fail-load', () => {
    console.log('Failed to load from file protocol, trying localhost...');
    if (!isDev) {
      mainWindow.loadURL('http://localhost:3000');
    }
  });

  // Open DevTools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Create window when app is ready
app.on('ready', createWindow);

// Quit when all windows are closed
app.on('window-all-closed', () => {
  // On macOS, keep app active until user quits with Cmd+Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Re-create window when app is activated (macOS)
app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Handle any uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});
