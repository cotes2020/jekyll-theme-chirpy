import { TocMobile as mobile } from './toc/toc-mobile';
import { TocDesktop as desktop } from './toc/toc-desktop';

const desktopMode = matchMedia('(min-width: 1200px)');

function refresh(e) {
  if (e.matches) {
    if (mobile.popupOpened) {
      mobile.hidePopup();
    }

    desktop.refresh();
  } else {
    mobile.refresh();
  }
}

function init() {
  if (document.querySelector('main>article[data-toc="true"]') === null) {
    return;
  }

  // Avoid create multiple instances of Tocbot. Ref: <https://github.com/tscanlin/tocbot/issues/203>
  if (desktopMode.matches) {
    desktop.init();
  } else {
    mobile.init();
  }

  desktopMode.onchange = refresh;
}

export { init as initToc };
