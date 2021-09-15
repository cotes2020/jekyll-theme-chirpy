/*
 * Initial the clipboard.js object
 *
 * Dependencies:
 *   - popper.js (https://github.com/popperjs/popper-core)
 *   - clipboard.js (https://github.com/zenorocha/clipboard.js)
 */

$(function() {
  const btnSelector = '.code-header>button';
  const ICON_DEFAULT = getIcon(btnSelector);
  const ICON_SUCCESS = 'fas fa-check';
  const ATTR_LOCKED = 'locked';
  const TIMEOUT = 2000; // in milliseconds

  const clipboard = new ClipboardJS(btnSelector, {
    target(trigger) {
      return trigger.parentNode.nextElementSibling;
    }
  });

  $(btnSelector).tooltip({
    trigger: 'click',
    placement: 'left'
  });

  function setTooltip(btn, msg) {
    $(btn).tooltip('hide')
      .attr('data-original-title', msg)
      .tooltip('show');
  }

  function hideTooltip(btn) {
    setTimeout(function() {
      $(btn).tooltip('hide');
    }, TIMEOUT);
  }

  function getIcon(btn) {
    let iconNode = $(btn).children();
    return iconNode.attr('class');;
  }

  function setSuccessIcon(btn) {
    let btnNode = $(btn);
    let iconNode = btnNode.children();
    btnNode.attr(ATTR_LOCKED, true);
    iconNode.attr('class', ICON_SUCCESS);
  }

  function resumeIcon(btn) {
    let btnNode = $(btn);
    let iconNode = btnNode.children();

    setTimeout(function() {
      btnNode.removeAttr(ATTR_LOCKED);
      iconNode.attr('class', ICON_DEFAULT);
    }, TIMEOUT);
  }

  function isLocked(btn) {
    let locked = $(btn).attr(ATTR_LOCKED);
    return locked === 'true';
  }

  clipboard.on('success', (e) => {
    e.clearSelection();

    if (isLocked(e.trigger)) {
      return;
    }

    setTooltip(e.trigger, 'Copied!');
    hideTooltip(e.trigger);

    setSuccessIcon(e.trigger);
    resumeIcon($(e.trigger));

  });

});
