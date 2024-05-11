/**
 * Add listener for theme mode toggle
 */
const toggle = document.getElementById('mode-toggle');

export function modeWatcher() {
  if (!toggle) {
    return;
  }

  toggle.addEventListener('click', () => {
    modeToggle.flipMode();
  });
}
