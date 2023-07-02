import { modeWatcher } from '../components/mode-watcher';
import { sidebarExpand } from '../components/sidebar';

export function initSidebar() {
  modeWatcher();
  sidebarExpand();
}
