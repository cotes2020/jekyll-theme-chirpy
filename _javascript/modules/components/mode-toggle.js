/**
 * Add listener for theme mode toggle
 */

export function modeWatcher() {
  const $toggle = document.getElementById('mode-toggle');
  console.log('Mode watcher initialized');
  if (!$toggle) {
    return;
  }

  $toggle.addEventListener('click', () => {
    Theme.flip();
  });
}
