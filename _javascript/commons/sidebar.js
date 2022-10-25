/**
 * Expand or close the sidebar in mobile screens.
 */

$(function () {
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
