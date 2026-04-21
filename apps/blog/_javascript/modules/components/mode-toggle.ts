/**
 * Add listener for theme mode toggle
 */

import Theme from '../../theme';

export function modeWatcher(): void {
  const $toggle = document.getElementById('mode-toggle');

  if (!$toggle) {
    return;
  }

  $toggle.addEventListener('click', () => {
    Theme.flip();
  });
}
