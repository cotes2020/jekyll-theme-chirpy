import { basic, initSidebar, initTopbar } from './modules/layouts';
import { loadImg } from './modules/components/img-loading';
import { imgPopup } from './modules/components/img-popup';
import { initClipboard } from './modules/components/clipboard';
import { loadMermaid } from './modules/components/mermaid';

loadImg();
imgPopup();
initSidebar();
initTopbar();
initClipboard();
loadMermaid();
basic();
