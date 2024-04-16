import { basic, initSidebar, initTopbar } from './modules/layouts';
import {loadImg, imgPopup, initClipboard, toc} from './modules/plugins';

basic();
initSidebar();
initTopbar();
loadImg();
imgPopup();
initClipboard();
toc();
