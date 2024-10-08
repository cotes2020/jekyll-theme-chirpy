/*
  TOC topbar and popup for mobile/tablet
 */

const tocBar = document.getElementById('toc-bar');
const soloTrigger = document.getElementById('toc-solo-trigger');
const triggers = document.getElementsByClassName('toc-trigger');

const popup = document.getElementById('toc-popup');
const btnClose = document.getElementById('toc-popup-close');

export class TocMobile {
  static isVisible = false;
  static FROZEN = 'overflow-hidden';
  static barHeight = 16 * 3; // 3rem

  static options = {
    tocSelector: '#toc-popup-content',
    contentSelector: '.content',
    ignoreSelector: '[data-toc-skip]',
    headingSelector: 'h2, h3, h4',
    orderedList: false,
    scrollSmooth: false,
    collapseDepth: 4,
    headingsOffset: this.barHeight
  };

  static initBar() {
    if (tocBar === null) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          tocBar.classList.toggle('invisible', entry.isIntersecting);
        });
      },
      { rootMargin: `-${this.barHeight}px 0px 0px 0px` }
    );

    observer.observe(soloTrigger);
  }

  static refresh() {
    if (!this.isVisible) {
      this.initComponents();
    }

    tocbot.refresh(this.options);
  }

  static showPopup() {
    TocMobile.setScrollEnabled(false);
    popup.showModal();
  }

  static hidePopup() {
    TocMobile.setScrollEnabled(true);
    popup.close();
  }

  static setScrollEnabled(enabled) {
    document.documentElement.classList.toggle(this.FROZEN, !enabled);
    document.body.classList.toggle(this.FROZEN, !enabled);
  }

  static initComponents() {
    this.initBar();

    [...triggers].forEach((trigger) => {
      trigger.addEventListener('click', this.showPopup);
    });

    popup?.addEventListener('click', this.hidePopup);
    popup?.addEventListener('cancel', this.hidePopup);
    btnClose?.addEventListener('click', this.hidePopup);

    this.isVisible = true;
  }

  static init() {
    tocbot.init(this.options);
    this.initComponents();
  }
}
