import { basic, initSidebar, initTopbar } from './modules/layouts';
import {
  imgExtra,
  initLocaleDatetime,
  initClipboard,
  smoothScroll,
  initPageviews
} from './modules/plugins';

basic();
initSidebar();
initTopbar();
imgExtra();
initLocaleDatetime();
initClipboard();
smoothScroll();
initPageviews();
