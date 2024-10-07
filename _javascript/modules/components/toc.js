import { TocMobile as mobile } from './toc/toc-mobile';
import { TocDesktop as desktop } from './toc/toc-desktop';

const mediaQuery = matchMedia('(min-width: 1200px)');

function refresh() {
  if (mediaQuery.matches) {
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
  if (mediaQuery.matches) {
    desktop.init();
  } else {
    mobile.init();
  }

  mediaQuery.addEventListener('change', refresh);
}

export { init as initToc };
