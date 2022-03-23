/**
 * Calculate the Timeago
 *
 * Requirement: <https://github.com/iamkun/dayjs>
 */

$(function() {
  const attrTimestamp = LocaleHelper.attrTimestamp();
  const attrCapitalize = 'data-capitalize';
  const $timeago = $(".timeago");

  let timeagoTasks = $timeago.length;
  let intervalId = void 0;

  dayjs.locale(LocaleHelper.locale());
  dayjs.extend(window.dayjs_plugin_relativeTime);
  dayjs.extend(window.dayjs_plugin_localizedFormat);

  function relativetime($elem) {
    const now = dayjs();
    const past = dayjs.unix(LocaleHelper.getTimestamp($elem));

    let diffMonth = now.diff(past, 'month', true);
    if (diffMonth > 10) { // year ago range: 11 months to 17months
      $elem.removeAttr(attrTimestamp);
      return past.format('ll'); // see: https://day.js.org/docs/en/display/format#list-of-localized-formats
    }

    let diffMinute = now.diff(past, 'minute', true);
    if (diffMinute > 44) { // an hour ago range: 45 to 89 minutes
      $elem.removeAttr(attrTimestamp);
    }

    return past.fromNow();
  }

  function updateTimeago() {
    $timeago.each(function() {
      if (typeof $(this).attr(attrTimestamp) === 'undefined') {
        timeagoTasks -= 1;
        return;
      }

      let relativeTime = relativetime($(this));
      const capitalize = $(this).attr(attrCapitalize);
      if (typeof capitalize !== 'undefined' && capitalize === 'true') {
        relativeTime = relativeTime.replace(/^\w/, (c) => c.toUpperCase());
      }

      if ($(this).text() !== relativeTime) {
        $(this).text(relativeTime);
      }
    });

    if (timeagoTasks === 0 && typeof intervalId !== "undefined") {
      clearInterval(intervalId); /* stop interval */
    }

    return timeagoTasks;
  }

  function setupTooltips() {
    $timeago.each(function() {
      const tooltip = $(this).attr('data-toggle');
      if (typeof tooltip === 'undefined' || tooltip !== 'tooltip') {
        return;
      }

      const df = $(this).attr('data-tooltip-df');
      const ts = LocaleHelper.getTimestamp($(this));
      const dateStr = dayjs.unix(ts).format(df);
      $(this).attr('data-original-title', dateStr);
      $(this).removeAttr('data-tooltip-df');
    });
  }

  if (timeagoTasks === 0) {
    return;
  }

  setupTooltips();

  if (updateTimeago()) { /* run immediately */
    intervalId = setInterval(updateTimeago, 60 * 1000); /* run every minute */
  }

});
