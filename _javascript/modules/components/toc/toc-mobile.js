/**
 * TOC button, topbar and popup for mobile devices
 */

const $tocBar = document.getElementById('toc-bar');
const $soloTrigger = document.getElementById('toc-solo-trigger');
const $triggers = document.getElementsByClassName('toc-trigger');
const $popup = document.getElementById('toc-popup');
const $btnClose = document.getElementById('toc-popup-close');

const SCROLL_LOCK = 'overflow-hidden';
const CLOSING = 'closing';

export class TocMobile {
  static #invisible = true;
  static #barHeight = 16 * 3; // 3rem

  static options = {
    tocSelector: '#toc-popup-content',
    contentSelector: '.content',
    ignoreSelector: '[data-toc-skip]',
    headingSelector: 'h2, h3, h4',
    orderedList: false,
    scrollSmooth: false,
    collapseDepth: 4,
    headingsOffset: this.#barHeight
  };

  static initBar() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          $tocBar.classList.toggle('invisible', entry.isIntersecting);
        });
      },
      { rootMargin: `-${this.#barHeight}px 0px 0px 0px` }
    );

    observer.observe($soloTrigger);
    this.#invisible = false;
  }

  static listenAnchors() {
    const $anchors = document.getElementsByClassName('toc-link');
    [...$anchors].forEach((anchor) => {
      anchor.onclick = () => this.hidePopup();
    });
  }

  static refresh() {
    if (this.#invisible) {
      this.initComponents();
    }
    tocbot.refresh(this.options);
    this.listenAnchors();
  }

  static get popupOpened() {
    return $popup.open;
  }

  static showPopup() {
    this.lockScroll(true);
    $popup.showModal();
    const activeItem = $popup.querySelector('li.is-active-li');
    activeItem.scrollIntoView({ block: 'center' });
  }

  static hidePopup() {
    $popup.toggleAttribute(CLOSING);

    $popup.addEventListener(
      'animationend',
      () => {
        $popup.toggleAttribute(CLOSING);
        $popup.close();
      },
      { once: true }
    );

    this.lockScroll(false);
  }

  static lockScroll(enable) {
    document.documentElement.classList.toggle(SCROLL_LOCK, enable);
    document.body.classList.toggle(SCROLL_LOCK, enable);
  }

  static clickBackdrop(event) {
    if ($popup.hasAttribute(CLOSING)) {
      return;
    }

    const rect = event.target.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      this.hidePopup();
    }
  }

  static initComponents() {
    this.initBar();

    [...$triggers].forEach((trigger) => {
      trigger.onclick = () => this.showPopup();
    });

    $popup.onclick = (e) => this.clickBackdrop(e);
    $btnClose.onclick = () => this.hidePopup();
    $popup.oncancel = (e) => {
      e.preventDefault();
      this.hidePopup();
    };
  }

  static init() {
    tocbot.init(this.options);
    this.listenAnchors();
    this.initComponents();
  }
}
