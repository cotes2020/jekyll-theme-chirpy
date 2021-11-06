/*
 * Clipboard functions
 *
 * Dependencies:
 *   - popper.js (https://github.com/popperjs/popper-core)
 *   - clipboard.js (https://github.com/zenorocha/clipboard.js)
 */

$(function() {
  const btnSelector = '.code-header>button';
  const ICON_SUCCESS = 'fas fa-check';
  const ATTR_TIMEOUT = 'timeout';
  const TIMEOUT = 2000; // in milliseconds

  function isLocked(node) {
    if ($(node)[0].hasAttribute(ATTR_TIMEOUT)) {
      let timeout = $(node).attr(ATTR_TIMEOUT);
      if (Number(timeout) > Date.now()) {
        return true;
      }
    }
    return false;
  }

  function lock(node) {
    $(node).attr(ATTR_TIMEOUT, Date.now() + TIMEOUT);
  }

  function unlock(node) {
    $(node).removeAttr(ATTR_TIMEOUT);
  }

  /* --- Copy code block --- */

  // Initial the clipboard.js object
  const clipboard = new ClipboardJS(btnSelector, {
    target(trigger) {
      let codeBlock = trigger.parentNode.nextElementSibling;
      return codeBlock.querySelector('code .rouge-code');
    }
  });

  $(btnSelector).tooltip({
    trigger: 'hover',
    placement: 'left'
  });

  function getIcon(btn) {
    let iconNode = $(btn).children();
    return iconNode.attr('class');
  }

  const ICON_DEFAULT = getIcon(btnSelector);

  function showTooltip(btn) {
    const succeedTitle = $(btn).attr('title-succeed');
    $(btn).attr('data-original-title', succeedTitle).tooltip('show');
  }

  function hideTooltip(btn) {
    $(btn).tooltip('hide').removeAttr('data-original-title');
  }

  function setSuccessIcon(btn) {
    let btnNode = $(btn);
    let iconNode = btnNode.children();
    iconNode.attr('class', ICON_SUCCESS);
  }

  function resumeIcon(btn) {
    let btnNode = $(btn);
    let iconNode = btnNode.children();
    iconNode.attr('class', ICON_DEFAULT);
  }

  clipboard.on('success', (e) => {
    e.clearSelection();

    const trigger = e.trigger;
    if (isLocked(trigger)) {
      return;
    }

    setSuccessIcon(trigger);
    showTooltip(trigger);
    lock(trigger);

    setTimeout(() => {
      hideTooltip(trigger);
      resumeIcon(trigger);
      unlock(trigger);
    }, TIMEOUT);

  });

  /* --- Post link sharing --- */

  $('#copy-link').click((e) => {

    let target = $(e.target);

    if (isLocked(target)) {
      return;
    }

    // Copy URL to clipboard

    const url = window.location.href;
    const $temp = $("<input>");

    $("body").append($temp);
    $temp.val(url).select();
    document.execCommand("copy");
    $temp.remove();

    // Switch tooltip title

    const defaultTitle = target.attr('data-original-title');
    const succeedTitle = target.attr('title-succeed');

    target.attr('data-original-title', succeedTitle).tooltip('show');
    lock(target);

    setTimeout(() => {
      target.attr('data-original-title', defaultTitle);
      unlock(target);
    }, TIMEOUT);

  });

});
