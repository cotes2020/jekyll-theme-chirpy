/**
 * Set up image popup
 *
 * Dependencies: https://github.com/biati-digital/glightbox
 */

import { GLightbox, type GLightboxInstance } from '../globals-glightbox';
import { Theme } from '../globals-theme';

const lightImages = '.popup:not(.dark)';
const darkImages = '.popup:not(.light)';
let selector = lightImages;

function swapImages(
  current: GLightboxInstance,
  reverse: GLightboxInstance | null
): [GLightboxInstance, GLightboxInstance] {
  if (selector === lightImages) {
    selector = darkImages;
  } else {
    selector = lightImages;
  }

  if (reverse === null) {
    reverse = GLightbox({ selector: `${selector}` });
  }

  return [reverse, current];
}

export function imgPopup(): void {
  if (document.querySelector('.popup') === null) {
    return;
  }

  const hasDualImages = !(
    document.querySelector('.popup.light') === null &&
    document.querySelector('.popup.dark') === null
  );

  if (Theme.visualState === Theme.DARK) {
    selector = darkImages;
  }

  let current: GLightboxInstance = GLightbox({ selector: `${selector}` });

  if (hasDualImages && Theme.switchable) {
    let reverse: GLightboxInstance | null = null;

    window.addEventListener('message', (event) => {
      if (event.source === window && event.data && event.data.id === Theme.ID) {
        [current, reverse] = swapImages(current, reverse);
      }
    });
  }
}
