import { basic, initSidebar, initTopbar } from './modules/layouts';
import { loadImg, imgPopup, initClipboard } from './modules/plugins';

loadImg();
imgPopup();
initSidebar();
initTopbar();
initClipboard();
basic();
