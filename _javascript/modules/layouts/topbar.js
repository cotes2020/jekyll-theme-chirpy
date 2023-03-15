import { convertTitle } from '../components/convert-title';
import { displaySearch } from '../components/search-display';
import { switchTopbar } from '../components/topbar-switcher';

export function initTopbar() {
  convertTitle();
  displaySearch();
  switchTopbar();
}
