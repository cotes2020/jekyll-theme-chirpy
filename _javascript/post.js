import { basic } from './modules/layouts/basic';
import { initSidebar } from './modules/layouts/sidebar';
import { initTopbar } from './modules/layouts/topbar';
import { loadImg } from './modules/components/img-loading';
import { imgPopup } from './modules/components/img-popup';
import { initLocaleDatetime } from './modules/components/locale-datetime';
import { initClipboard } from './modules/components/clipboard';
import { toc } from './modules/components/toc';

loadImg();
toc();
imgPopup();
initSidebar();
initLocaleDatetime();
initClipboard();
initTopbar();
basic();
