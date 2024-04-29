import { basic, initSidebar, initTopbar } from './modules/layouts';
import { initLocaleDatetime, loadImg } from './modules/plugins';

loadImg();
initLocaleDatetime();
initSidebar();
initTopbar();
basic();
