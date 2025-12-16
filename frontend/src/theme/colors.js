/**
 * Color Palette & Theme Configuration
 * Centralized theme for consistent styling across the application
 */

export const COLORS = {
  // Primary Colors
  primary: {
    light: '#0ea5e9',    // sky-500
    main: '#0284c7',     // sky-600
    dark: '#0369a1',     // sky-700
    darker: '#02558a',   // sky-800
  },

  // Secondary Colors (Accent)
  secondary: {
    light: '#06b6d4',    // cyan-500
    main: '#0891b2',     // cyan-600
    dark: '#0e7490',     // cyan-700
  },

  // Status Colors
  success: {
    light: '#10b981',    // emerald-500
    main: '#059669',     // emerald-600
    dark: '#047857',     // emerald-700
  },

  warning: {
    light: '#f59e0b',    // amber-500
    main: '#d97706',     // amber-600
    dark: '#b45309',     // amber-700
  },

  error: {
    light: '#ef4444',    // red-500
    main: '#dc2626',     // red-600
    dark: '#b91c1c',     // red-700
  },

  info: {
    light: '#3b82f6',    // blue-500
    main: '#2563eb',     // blue-600
    dark: '#1d4ed8',     // blue-700
  },

  // Neutral/Background Colors
  neutral: {
    0: '#ffffff',        // white
    50: '#f9fafb',       // gray-50
    100: '#f3f4f6',      // gray-100
    200: '#e5e7eb',      // gray-200
    300: '#d1d5db',      // gray-300
    400: '#9ca3af',      // gray-400
    500: '#6b7280',      // gray-500
    600: '#4b5563',      // gray-600
    700: '#374151',      // gray-700
    800: '#1f2937',      // gray-800
    900: '#111827',      // gray-900
  },

  // Dark Mode Backgrounds
  background: {
    surface: '#0f172a',        // slate-900
    surfaceLight: '#1e293b',   // slate-800
    surfaceLighter: '#334155',  // slate-700
    overlay: 'rgba(0, 0, 0, 0.5)',
  },

  // Gradient Backgrounds
  gradient: {
    primary: 'from-sky-500 to-cyan-500',
    dark: 'from-slate-900 via-slate-800 to-slate-900',
    accent: 'from-purple-600 to-blue-600',
  },

  // Text Colors
  text: {
    primary: '#ffffff',
    secondary: '#d1d5db',      // gray-300
    tertiary: '#9ca3af',       // gray-400
    inverse: '#111827',        // gray-900
  },

  // Border Colors
  border: {
    light: 'rgba(203, 213, 225, 0.1)',     // slate-300 with opacity
    main: '#475569',                        // slate-600
    dark: '#334155',                        // slate-700
  },
};

/**
 * Tailwind Class Combinations for Common Patterns
 */
export const STYLES = {
  // Buttons
  button: {
    primary: 'bg-sky-600 hover:bg-sky-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200',
    secondary: 'bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200',
    ghost: 'hover:bg-slate-700 text-slate-300 hover:text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200',
    danger: 'bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200',
    small: 'text-sm font-medium py-1 px-3 rounded transition-colors duration-200',
  },

  // Cards
  card: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-4',
  cardHover: 'bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-4 hover:bg-slate-700/50 hover:border-cyan-500/50 transition-all duration-200',

  // Input Fields
  input: 'w-full px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition-colors duration-200',

  // Text
  heading: {
    h1: 'text-4xl font-bold text-white',
    h2: 'text-2xl font-semibold text-white',
    h3: 'text-lg font-semibold text-white',
    h4: 'text-base font-semibold text-white',
  },

  // Badges
  badge: {
    primary: 'inline-block px-3 py-1 bg-sky-600/20 text-sky-300 text-xs font-semibold rounded-full border border-sky-600/50',
    success: 'inline-block px-3 py-1 bg-emerald-600/20 text-emerald-300 text-xs font-semibold rounded-full border border-emerald-600/50',
    warning: 'inline-block px-3 py-1 bg-amber-600/20 text-amber-300 text-xs font-semibold rounded-full border border-amber-600/50',
    error: 'inline-block px-3 py-1 bg-red-600/20 text-red-300 text-xs font-semibold rounded-full border border-red-600/50',
  },

  // Messages
  messageUser: 'bg-sky-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-xs lg:max-w-md',
  messageAssistant: 'bg-slate-700 text-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-xs lg:max-w-md',

  // Sidebar/Navigation
  navItem: 'block w-full text-left px-4 py-3 rounded-lg hover:bg-slate-700 text-slate-300 hover:text-white transition-colors duration-200',
  navItemActive: 'block w-full text-left px-4 py-3 rounded-lg bg-sky-600 text-white transition-colors duration-200',
};

/**
 * Theme Hook for React Components
 */
export const useTheme = () => {
  return {
    colors: COLORS,
    styles: STYLES,
  };
};

export default COLORS;
