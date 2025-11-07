// Theme management
export const THEME_KEY = 'theme';
export const THEME_ATTR = 'data-mode';

export type Theme = 'light' | 'dark';

export function getThemePreference(): Theme {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'light' || stored === 'dark') {
    return stored;
  }

  // Check system preference
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function setTheme(theme: Theme) {
  document.documentElement.setAttribute(THEME_ATTR, theme);
  localStorage.setItem(THEME_KEY, theme);

  // Dispatch event for components that need to react to theme changes
  window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

export function toggleTheme() {
  const current = getThemePreference();
  const next = current === 'light' ? 'dark' : 'light';
  setTheme(next);
  return next;
}

export function initTheme() {
  const theme = getThemePreference();
  document.documentElement.setAttribute(THEME_ATTR, theme);
}

// Initialize theme on page load
if (typeof window !== 'undefined') {
  // Apply theme immediately to prevent flash
  initTheme();

  // Setup theme toggle
  document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('mode-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        toggleTheme();
      });
    }
  });

  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    // Only auto-switch if user hasn't set a preference
    if (!localStorage.getItem(THEME_KEY)) {
      setTheme(e.matches ? 'dark' : 'light');
    }
  });
}
