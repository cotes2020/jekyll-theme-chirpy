/**
 * A tool for smooth scrolling and topbar switcher
 */

const ATTR_TOPBAR_VISIBLE = 'data-topbar-visible';
const $body = $('body');
const $topbarWrapper = $('#topbar-wrapper');

export default class ScrollHelper {
  static scrollUpCount = 0; // the number of times the scroll up was triggered by ToC or anchor
  static topbarIsLocked = false;
  static orientationIsLocked = false;

  static hideTopbar() {
    $body.attr(ATTR_TOPBAR_VISIBLE, 'false');
  }

  static showTopbar() {
    $body.attr(ATTR_TOPBAR_VISIBLE, 'true');
  }

  // scroll up

  static addScrollUpTask() {
    ScrollHelper.scrollUpCount += 1;
    if (!ScrollHelper.topbarIsLocked) {
      ScrollHelper.topbarIsLocked = true;
    }
  }

  static popScrollUpTask() {
    ScrollHelper.scrollUpCount -= 1;
  }

  static hasScrollUpTask() {
    return ScrollHelper.scrollUpCount > 0;
  }

  static topbarLocked() {
    return ScrollHelper.topbarIsLocked === true;
  }

  static unlockTopbar() {
    ScrollHelper.topbarIsLocked = false;
  }

  static getTopbarHeight() {
    return $topbarWrapper.outerHeight();
  }

  // orientation change

  static orientationLocked() {
    return ScrollHelper.orientationIsLocked === true;
  }

  static lockOrientation() {
    ScrollHelper.orientationIsLocked = true;
  }

  static unLockOrientation() {
    ScrollHelper.orientationIsLocked = false;
  }
}
