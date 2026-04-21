import { basic, initTopbar, initSidebar } from './modules/layouts';

import { loadImg } from './modules/components/img-loading';
import { imgPopup } from './modules/components/img-popup';
import { initLocaleDatetime } from './modules/components/locale-datetime';
import { initClipboard } from './modules/components/clipboard';
import { initToc } from './modules/components/toc';
import { loadMermaid } from './modules/components/mermaid';

loadImg();
initToc();
imgPopup();
initSidebar();
initLocaleDatetime();
initClipboard();
initTopbar();
loadMermaid();
basic();
