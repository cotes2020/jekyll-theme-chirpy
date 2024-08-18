import { basic } from './modules/layouts/basic';
import { initSidebar } from './modules/layouts/sidebar';
import { initTopbar } from './modules/layouts/topbar';
import { initLocaleDatetime } from './modules/components/locale-datetime';

initSidebar();
initTopbar();
initLocaleDatetime();
basic();
