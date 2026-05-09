/**
 * TOC button, topbar and popup for mobile devices
 */

import { tocbot } from '../../globals-tocbot';

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

  static initBar(): void {
    if (!$tocBar || !$soloTrigger) return;
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

  static listenAnchors(): void {
    const $anchors = document.getElementsByClassName('toc-link');
    [...$anchors].forEach((anchor) => {
      (anchor as HTMLElement).onclick = () => this.hidePopup();
    });
  }

  static refresh(): void {
    if (this.#invisible) {
      this.initComponents();
    }
    tocbot.refresh(this.options);
    this.listenAnchors();
  }

  static get popupOpened(): boolean {
    return ($popup as HTMLDialogElement | null)?.open ?? false;
  }

  static showPopup(): void {
    if (!$popup) return;
    this.lockScroll(true);
    ($popup as HTMLDialogElement).showModal();
    const activeItem = $popup.querySelector<HTMLElement>('li.is-active-li');
    activeItem?.scrollIntoView({ block: 'center' });
  }

  static hidePopup(): void {
    if (!$popup) return;
    $popup.toggleAttribute(CLOSING);

    $popup.addEventListener(
      'animationend',
      () => {
        $popup.toggleAttribute(CLOSING);
        ($popup as HTMLDialogElement).close();
      },
      { once: true }
    );

    this.lockScroll(false);
  }

  static lockScroll(enable: boolean): void {
    document.documentElement.classList.toggle(SCROLL_LOCK, enable);
    document.body.classList.toggle(SCROLL_LOCK, enable);
  }

  static clickBackdrop(event: MouseEvent): void {
    if (!$popup || $popup.hasAttribute(CLOSING)) {
      return;
    }

    const rect = ($popup).getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      this.hidePopup();
    }
  }

  static initComponents(): void {
    this.initBar();

    [...$triggers].forEach((trigger) => {
      (trigger as HTMLElement).onclick = () => this.showPopup();
    });

    if (!$popup || !$btnClose) return;

    $popup.onclick = (e) => this.clickBackdrop(e);
    ($btnClose).onclick = () => this.hidePopup();
    ($popup as HTMLDialogElement).oncancel = (e) => {
      e.preventDefault();
      this.hidePopup();
    };
  }

  static init(): void {
    tocbot.init(this.options);
    this.listenAnchors();
    this.initComponents();
  }
}
