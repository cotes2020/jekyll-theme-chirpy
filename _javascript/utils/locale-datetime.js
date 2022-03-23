/**
 * Update month/day to locale datetime
 *
 * Requirement: <https://github.com/iamkun/dayjs>
 */

$(function() {
  dayjs.locale(LocaleHelper.locale());
  dayjs.extend(window.dayjs_plugin_localizedFormat);

  $(`[${LocaleHelper.attrTimestamp()}]`).each(function () {
    const date = dayjs.unix(LocaleHelper.getTimestamp($(this)));
    const df = LocaleHelper.getDateFormat($(this));
    const text = date.format(df);

    $(this).text(text);
    $(this).removeAttr(LocaleHelper.attrTimestamp());
    $(this).removeAttr(LocaleHelper.attrDateFormat());
  });
});
