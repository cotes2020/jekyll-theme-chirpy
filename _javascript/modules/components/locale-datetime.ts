/**
 * Update month/day to locale datetime
 *
 * Requirement: <https://github.com/iamkun/dayjs>
 */

import { dayjs, dayjsLocalizedFormatPlugin } from '../globals';

/* A tool for locale datetime */
class LocaleHelper {
  static get attrTimestamp() {
    return 'data-ts';
  }

  static get attrDateFormat() {
    return 'data-df';
  }

  static get locale() {
    const lang = document.documentElement.getAttribute('lang') ?? 'en';
    return lang.substring(0, 2);
  }

  static getTimestamp(elem: Element): number {
    return Number(elem.getAttribute(this.attrTimestamp)); // unix timestamp
  }

  static getDateFormat(elem: Element): string | undefined {
    return elem.getAttribute(this.attrDateFormat) ?? undefined;
  }
}

export function initLocaleDatetime(): void {
  dayjs.locale(LocaleHelper.locale);
  dayjs.extend(dayjsLocalizedFormatPlugin());

  document
    .querySelectorAll(`[${LocaleHelper.attrTimestamp}]`)
    .forEach((elem) => {
      const date = dayjs.unix(LocaleHelper.getTimestamp(elem));
      const text = date.format(LocaleHelper.getDateFormat(elem));
      elem.textContent = text;
      elem.removeAttribute(LocaleHelper.attrTimestamp);
      elem.removeAttribute(LocaleHelper.attrDateFormat);

      // setup tooltips
      if (
        elem.hasAttribute('data-bs-toggle') &&
        elem.getAttribute('data-bs-toggle') === 'tooltip'
      ) {
        // see: https://day.js.org/docs/en/display/format#list-of-localized-formats
        const tooltipText = date.format('llll');
        elem.setAttribute('data-bs-title', tooltipText);
      }
    });
}
