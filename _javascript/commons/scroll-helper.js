/**
 * A tool for smooth scrolling and topbar switcher
 */
const ScrollHelper = (function () {
  const $body = $("body");
  const ATTR_TOPBAR_VISIBLE = "data-topbar-visible";
  const topbarHeight = $("#topbar-wrapper").outerHeight();

  let scrollUpCount = 0; // the number of times the scroll up was triggered by ToC or anchor
  let topbarLocked = false;
  let orientationLocked = false;

  return {
    hideTopbar: () => $body.attr(ATTR_TOPBAR_VISIBLE, false),
    showTopbar: () => $body.attr(ATTR_TOPBAR_VISIBLE, true),

    // scroll up

    addScrollUpTask: () => {
      scrollUpCount += 1;
      if (!topbarLocked) { topbarLocked = true; }
    },
    popScrollUpTask: () => scrollUpCount -= 1,
    hasScrollUpTask: () => scrollUpCount > 0,
    topbarLocked: () => topbarLocked === true,
    unlockTopbar: () => topbarLocked = false,
    getTopbarHeight: () => topbarHeight,

    // orientation change

    orientationLocked: () => orientationLocked === true,
    lockOrientation: () => orientationLocked = true,
    unLockOrientation: () => orientationLocked = false
  };

}());
