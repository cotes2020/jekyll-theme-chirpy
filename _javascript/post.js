import { basic, initSidebar, initTopbar } from './modules/layouts';
import {
  loadImg,
  imgPopup,
  initLocaleDatetime,
  initClipboard,
  toc
} from './modules/plugins';

initSidebar();
initTopbar();
loadImg();
imgPopup();
initLocaleDatetime();
initClipboard();
toc();
basic();
