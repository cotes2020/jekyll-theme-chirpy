/**
 * Expand or close the sidebar in mobile screens.
 */

const ATTR_DISPLAY = 'sidebar-display';

class SidebarUtil {
  static isExpanded = false;

  static toggle() {
    if (SidebarUtil.isExpanded === false) {
      document.body.setAttribute(ATTR_DISPLAY, '');
    } else {
      document.body.removeAttribute(ATTR_DISPLAY);
    }

    SidebarUtil.isExpanded = !SidebarUtil.isExpanded;
  }
}

export function sidebarExpand() {
  document
    .getElementById('sidebar-trigger')
    .addEventListener('click', SidebarUtil.toggle);

  document.getElementById('mask').addEventListener('click', SidebarUtil.toggle);
}
