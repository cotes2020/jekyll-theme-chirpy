/**
 * Expand or close the sidebar in mobile screens.
 * v2.0
 * https://github.com/cotes2020/jekyll-theme-chirpy
 * © 2018-2019 Cotes Chung
 * MIT License
 */

$(function() {

  const sidebarUtil = (function () {
    const ATTR_DISPLAY = "sidebar-display";
    let isExpanded = false;
    const body = $("body");

    return {
      toggle() {
        if (isExpanded === false) {
          body.attr(ATTR_DISPLAY, "");
        } else {
          body.removeAttr(ATTR_DISPLAY);
        }

        isExpanded = !isExpanded;
      }
    };

  }());

  $("#sidebar-trigger").click(sidebarUtil.toggle);

  $("#mask").click(sidebarUtil.toggle);

});
