import {
  basic,
  initOverlay,
  initSidebar,
  initTopbar,
  initPostTopbar
} from './modules/layouts';
import {
  loadImg,
  imgPopup,
  initLocaleDatetime,
  initClipboard,
  toc
} from './modules/plugins';

loadImg();
toc();
imgPopup();
initSidebar();
initLocaleDatetime();
initClipboard();
initTopbar();
initPostTopbar();
initOverlay();
basic();
