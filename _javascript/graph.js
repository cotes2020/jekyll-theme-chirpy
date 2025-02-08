import { basic, initSidebar, initTopbar } from './modules/layouts';
import { graphInit } from './modules/components/graph-visualization';

initSidebar();
initTopbar();
basic();

graphInit();