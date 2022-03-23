/**
 * A tool for locale datetime
 */

const LocaleHelper = (function () {
  const $preferLocale = $('meta[name="prefer-datetime-locale"]');
  const locale = $preferLocale.length > 0 ?
      $preferLocale.attr('content').toLowerCase() : $('html').attr('lang').substr(0, 2);
  const attrTimestamp = 'data-ts';
  const attrDateFormat = 'data-df';

  return {
    locale: () => locale,
    attrTimestamp: () => attrTimestamp,
    attrDateFormat: () => attrDateFormat,
    getTimestamp: ($elem) => Number($elem.attr(attrTimestamp)),  // unix timestamp
    getDateFormat: ($elem) => $elem.attr(attrDateFormat)
  };
}());
