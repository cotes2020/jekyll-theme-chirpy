/**
 * Set up image popup
 *
 * Dependencies: https://github.com/biati-digital/glightbox
 */

const html = document.documentElement;
const lightImages = '.popup:not(.dark)';
const darkImages = '.popup:not(.light)';
let selector = lightImages;

function updateImages(lightbox) {
  if (selector === lightImages) {
    selector = darkImages;
  } else {
    selector = lightImages;
  }

  lightbox.destroy();
  lightbox = GLightbox({ selector: `${selector}` });
}

export function imgPopup() {
  if (document.querySelector('.popup') === null) {
    return;
  }

  if (
    (html.hasAttribute('data-mode') &&
      html.getAttribute('data-mode') === 'dark') ||
    (!html.hasAttribute('data-mode') &&
      window.matchMedia('(prefers-color-scheme: dark)').matches)
  ) {
    selector = darkImages;
  }

  let lightbox = GLightbox({ selector: `${selector}` });

  if (document.getElementById('mode-toggle')) {
    window.addEventListener('message', (event) => {
      if (
        event.source === window &&
        event.data &&
        event.data.direction === ModeToggle.ID
      ) {
        updateImages(lightbox);
      }
    });
  }
}
