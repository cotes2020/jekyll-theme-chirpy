/**
 * Set up image popup
 *
 * Dependencies: https://github.com/biati-digital/glightbox
 */

const html = document.documentElement;
const lightImages = '.popup:not(.dark)';
const darkImages = '.popup:not(.light)';
let selector = lightImages;
let lightbox = GLightbox({ selector: `${selector}` });

if (
  (html.hasAttribute('data-mode') &&
    html.getAttribute('data-mode') === 'dark') ||
  (!html.hasAttribute('data-mode') &&
    window.matchMedia('(prefers-color-scheme: dark)').matches)
) {
  selector = darkImages;
}

function updateImages(event) {
  if (
    event.source === window &&
    event.data &&
    event.data.direction === ModeToggle.ID
  ) {
    if (selector === lightImages) {
      selector = darkImages;
    } else {
      selector = lightImages;
    }
  }

  lightbox.destroy();
  lightbox = GLightbox({ selector: `${selector}` });
}

export function imgPopup() {
  if (document.querySelector(`${selector}`) === null) {
    return;
  }

  if (document.getElementById('mode-toggle')) {
    window.addEventListener('message', updateImages);
  }
}
