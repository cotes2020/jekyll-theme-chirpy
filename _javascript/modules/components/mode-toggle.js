/**
 * Sets up the mode toggle dropdown, allowing users to switch between light, dark, and system themes.
 *
 * Dependencies:
 *  - Theme (${JS_ROOT}/theme.js)
 */

import 'bootstrap/js/src/dropdown.js';

const ACTIVE_CLASS = 'active';
const dropdown = document.querySelector('#mode-toggle + .dropdown-menu');
const activeMode = Theme.isSystemTheme
  ? Theme.Mode.SYSTEM
  : Theme.resolvedTheme;

export function modeWatcher() {
  if (!Theme.isToggleable) {
    return;
  }

  dropdown.querySelectorAll('.dropdown-item').forEach((option) => {
    const mode = option.dataset.themeMode;
    if (mode === activeMode) {
      option.classList.add(ACTIVE_CLASS);
      return;
    }
  });

  dropdown.addEventListener('click', (event) => {
    const current = event.target.closest('.dropdown-item');

    if (!current) {
      return;
    }

    const lastActive = dropdown.querySelector(`.${ACTIVE_CLASS}`);

    if (lastActive === current) {
      return;
    }

    lastActive.classList.remove(ACTIVE_CLASS);
    current.classList.add(ACTIVE_CLASS);
    Theme.update(current.dataset.themeMode);
  });
}
