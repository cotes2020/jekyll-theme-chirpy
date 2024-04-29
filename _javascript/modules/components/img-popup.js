/**
 * Set up image popup
 *
 * Dependencies: https://github.com/biati-digital/glightbox
 */

const IMG_CLASS = 'popup';

export function imgPopup() {
  if (document.getElementsByClassName(IMG_CLASS).length === 0) {
    return;
  }

  GLightbox({ selector: `.${IMG_CLASS}` });
}
