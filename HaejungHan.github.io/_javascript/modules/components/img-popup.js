/**
 * Set up image popup
 *
 * Dependencies: https://github.com/biati-digital/glightbox
 */

const html = document.documentElement;
const lightImages = '.popup:not(.dark)';
const darkImages = '.popup:not(.light)';
let selector = lightImages;

function updateImages(current, reverse) {
  if (selector === lightImages) {
    selector = darkImages;
  } else {
    selector = lightImages;
  }

  if (reverse === null) {
    reverse = GLightbox({ selector: `${selector}` });
  }

  [current, reverse] = [reverse, current];
}

export function imgPopup() {
  if (document.querySelector('.popup') === null) {
    return;
  }

  const hasDualImages = !(
    document.querySelector('.popup.light') === null &&
    document.querySelector('.popup.dark') === null
  );

  if (
    (html.hasAttribute('data-mode') &&
      html.getAttribute('data-mode') === 'dark') ||
    (!html.hasAttribute('data-mode') &&
      window.matchMedia('(prefers-color-scheme: dark)').matches)
  ) {
    selector = darkImages;
  }

  let current = GLightbox({ selector: `${selector}` });

  if (hasDualImages && document.getElementById('mode-toggle')) {
    let reverse = null;

    window.addEventListener('message', (event) => {
      if (
        event.source === window &&
        event.data &&
        event.data.direction === ModeToggle.ID
      ) {
        updateImages(current, reverse);
      }
    });
  }
}
