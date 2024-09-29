/*
 * Post topbar functions
 */

const postTopbar = document.getElementById('post-topbar');

const overlayTrigger = document.getElementById('overlay-trigger');
const overlayInlineTrigger = document.getElementById('overlay-inline-trigger');

const overlay = document.getElementById('overlay');
const overlayContent = document.getElementById('overlay-content');
const overlayClose = document.getElementById('overlay-close');

const UNLOADED = 'd-none';

function initPostTopbar() {
  window.addEventListener('scroll', () => {
    if (window.scrollY >= overlayInlineTrigger?.offsetTop) {
      postTopbar.classList.remove(UNLOADED);
    } else {
      postTopbar.classList.add(UNLOADED);
    }
  });
}

class Overlay {
  static NOSCROLL = 'overflow-hidden';

  static show() {
    this.setScrollEnabled(false);
    overlay.showModal();
  }

  static hide() {
    this.setScrollEnabled(true);
    overlay.close();
  }

  static setScrollEnabled(enabled) {
    if (enabled) {
      document.documentElement.classList.remove(this.NOSCROLL);
      document.body.classList.remove(this.NOSCROLL);
    } else {
      document.documentElement.classList.add(this.NOSCROLL);
      document.body.classList.add(this.NOSCROLL);
    }
  }

  static addTocToOverlay() {
    const toc = document.getElementById('toc-wrapper');
    const clonedToc = toc.cloneNode(true);
    this.removeContent();
    overlayContent.appendChild(clonedToc);
  }

  static removeContent() {
    while (overlayContent.firstChild) {
      overlayContent.removeChild(overlayContent.firstChild);
    }
  }

  static init() {
    [overlayTrigger, overlayInlineTrigger].forEach((e) =>
      e.addEventListener('click', () => {
        this.addTocToOverlay();
        this.show();
      })
    );

    overlay?.addEventListener('click', () => {
      this.hide();
      this.removeContent();
    });

    overlayClose?.addEventListener('click', () => {
      this.hide();
      this.removeContent();
    });
  }
}

export { initPostTopbar };

export function initOverlay() {
  Overlay.init();
}
