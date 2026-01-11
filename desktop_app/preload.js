const { contextBridge } = require('electron');

// Expose a safe API to the renderer process
contextBridge.exposeInMainWorld('electron', {
  isElectron: true,
  platform: process.platform,
  arch: process.arch,
});

// Forward all console messages to main process for debugging
console.log('Preload script loaded');
