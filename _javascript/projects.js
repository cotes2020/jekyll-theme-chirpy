import {basic, initSidebar, initTopbar} from './modules/layouts';
import {initLocaleDatetime, loadImg, toc} from './modules/plugins';

basic();
initSidebar();
initTopbar();
initLocaleDatetime();
toc();
loadImg();
