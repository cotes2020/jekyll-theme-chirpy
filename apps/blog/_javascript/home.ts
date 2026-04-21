import { basic, initSidebar, initTopbar } from './modules/layouts';
import { loadImg } from './modules/components/img-loading';
import { initLocaleDatetime } from './modules/components/locale-datetime';

loadImg();
initLocaleDatetime();
initSidebar();
initTopbar();
basic();
