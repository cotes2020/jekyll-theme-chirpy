import { basic } from './modules/layouts/basic';
import { initSidebar } from './modules/layouts/sidebar';
import { initTopbar } from './modules/layouts/topbar';
import { loadImg } from './modules/components/img-loading';
import { initLocaleDatetime } from './modules/components/locale-datetime';

loadImg();
initLocaleDatetime();
initSidebar();
initTopbar();
basic();
