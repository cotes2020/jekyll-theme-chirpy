import { basic } from './modules/layouts/basic';
import { initSidebar } from './modules/layouts/sidebar';
import { initTopbar } from './modules/layouts/topbar';
import { loadImg } from './modules/components/img-loading';
import { imgPopup } from './modules/components/img-popup';
import { initClipboard } from './modules/components/clipboard';

loadImg();
imgPopup();
initSidebar();
initTopbar();
initClipboard();
basic();
