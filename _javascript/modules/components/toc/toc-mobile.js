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
    headingsOffset: TocMobile.barHeight
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
      { rootMargin: `-${TocMobile.barHeight}px 0px 0px 0px` }
    );

    observer.observe(soloTrigger);
  }

  static refresh() {
    if (!TocMobile.isVisible) {
      TocMobile.initComponents();
    }

    tocbot.refresh(this.options);
  }

  static show() {
    TocMobile.setScrollEnabled(false);
    popup.showModal();
  }

  static hide() {
    TocMobile.setScrollEnabled(true);
    popup.close();
  }

  static setScrollEnabled(enabled) {
    document.documentElement.classList.toggle(this.FROZEN, !enabled);
    document.body.classList.toggle(this.FROZEN, !enabled);
  }

  static initComponents() {
    TocMobile.initBar();

    [...triggers].forEach((trigger) => {
      trigger.addEventListener('click', TocMobile.show);
    });

    popup?.addEventListener('click', TocMobile.hide);
    btnClose?.addEventListener('click', TocMobile.hide);

    TocMobile.isVisible = true;
  }

  static init() {
    tocbot.init(this.options);
    TocMobile.initComponents();
  }
}
