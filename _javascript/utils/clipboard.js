/*
 * Initial the clipboard.js object, see: <https://github.com/zenorocha/clipboard.js>
 *
 * Dependencies:
 *   - popper.js (https://github.com/popperjs/popper-core)
 *   - clipboard.js (https://github.com/zenorocha/clipboard.js)
 */
$(function() {
  const btnSelector = '.code-header>button';

  var clipboard = new ClipboardJS(btnSelector, {
    target(trigger) {
      return trigger.parentNode.nextElementSibling;
    }
  });

  function setTooltip(btn, msg) {
    $(btn).tooltip('hide')
      .attr('data-original-title', msg)
      .tooltip('show');
  }

  function hideTooltip(btn) {
    setTimeout(function() {
      $(btn).tooltip('hide');
    }, 1000);
  }

  $(btnSelector).tooltip({
    trigger: 'click',
    placement: 'left'
  });

  clipboard.on('success', function(e) {
    e.clearSelection();
    setTooltip(e.trigger, 'Copied!');
    hideTooltip(e.trigger);
  });

});
