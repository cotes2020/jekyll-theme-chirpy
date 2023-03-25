import { basic, initSidebar, initTopbar } from './modules/layouts';
import {
  imgExtra,
  initLocaleDatetime,
  initClipboard,
  smoothScroll,
  initPageviews,
  toc
} from './modules/plugins';

basic();
initSidebar();
initTopbar();
imgExtra();
initLocaleDatetime();
initClipboard();
toc();
smoothScroll(); // must be called after toc is created
initPageviews();
