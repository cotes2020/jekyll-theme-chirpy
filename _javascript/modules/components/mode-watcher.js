/**
 * Add listener for theme mode toggle
 */
const toggle = document.getElementById('mode-toggle');
const vcrToggle = document.getElementById('vcr-toggle');

export function modeWatcher() {
  if (!toggle && !vcrToggle) {
    return;
  }

  toggle.addEventListener('click', () => {
    modeToggle.flipMode();
  });
  vcrToggle.addEventListener('click', () => {
    vcr.flipMode();
  });
}
