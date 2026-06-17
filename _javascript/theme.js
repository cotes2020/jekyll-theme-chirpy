/**
 *  A utility class that manages the site's theme mode.
 *
 * Concepts:
 *  - Mode: dark, light, or system. The latter follows the operating system's preference.
 *  - Theme: The actual theme applied to the DOM, either dark or light. Determined by the mode or system preference.
 */
class Theme {
  /** @type {string} LocalStorage key for the selected theme mode. */
  static #storageKey = 'theme';

  static Mode = Object.freeze({
    DARK: 'dark',
    LIGHT: 'light',
    SYSTEM: 'system'
  });

  static #root = document.documentElement;

  /** @type {MediaQueryList} System dark-mode preference query. */
  static #mediaDark = window.matchMedia('(prefers-color-scheme: dark)');

  /** @returns {string|null} The theme currently set on the DOM. */
  static get #domTheme() {
    return this.#root.dataset.bsTheme || null;
  }

  /** @returns {string|null} The theme stored on the client. */
  static get #storedTheme() {
    return localStorage.getItem(this.#storageKey);
  }

  /** @returns {string} The theme preferred by the operating system. */
  static get #systemTheme() {
    return this.#prefersDark ? this.Mode.DARK : this.Mode.LIGHT;
  }

  /** @returns {boolean} Whether the operating system prefers dark mode. */
  static get #prefersDark() {
    return this.#mediaDark.matches;
  }

  /**
   * Applies a theme and optionally persists it as a user preference.
   *
   * @param {'light'|'dark'} theme
   * @param {{ persist?: boolean, domPersist?: boolean }} [options]
   *        - `persist`: Whether the theme is persisted in localStorage.
   *        - `domPersist`: Whether the theme is persisted in data attributes on the DOM.
   */
  static #apply(theme, { persist = false, domPersist = false } = {}) {
    this.#root.dataset.bsTheme = theme;

    if (persist) {
      localStorage.setItem(this.#storageKey, theme);
    }

    if (domPersist || persist) {
      this.#root.toggleAttribute('data-theme-persisted', true);
    }
  }

  /** Removes the stored user preference. */
  static #clearStorage() {
    localStorage.removeItem(this.#storageKey);
    this.#root.toggleAttribute('data-theme-persisted', false);
  }

  /** Broadcasts a theme change event to dependent modules. */
  static #notify() {
    window.postMessage({ id: this.eventId }, '*');
  }

  /** @type {boolean} Whether the current page allows theme toggling. */
  static isToggleable = this.#domTheme === null;

  static eventId = 'theme-updated';

  /** @returns {string} Resolved theme, falling back to the system preference. */
  static get resolvedTheme() {
    return this.#storedTheme || this.#systemTheme;
  }

  /** @returns {boolean} Whether the theme is determined by the system preference. */
  static get isSystemTheme() {
    return this.#storedTheme === null;
  }

  /** @returns {boolean} Whether the resolved theme is dark. */
  static get isDark() {
    return this.resolvedTheme === this.Mode.DARK;
  }

  /**
   * Creates a mode-indexed value map.
   *
   * @template T
   * @param {T} light Value for light mode.
   * @param {T} dark Value for dark mode.
   * @returns {{ light: T, dark: T }}
   */
  static newThemeMap(light, dark) {
    return {
      [this.Mode.LIGHT]: light,
      [this.Mode.DARK]: dark
    };
  }

  /** Initializes the theme from the stored value or system preference. */
  static init() {
    if (!this.isToggleable) {
      this.#clearStorage();
      return;
    }

    const storedTheme = this.#storedTheme;

    if (storedTheme) {
      this.#apply(storedTheme, { domPersist: true });
    } else {
      this.#apply(this.#systemTheme);
    }

    this.#mediaDark.addEventListener('change', () => {
      if (this.#storedTheme) {
        return;
      }

      this.#apply(this.#systemTheme);
      this.#notify();
    });
  }

  /**
   * Updates the theme by the specified mode.
   *
   * @param {'light'|'dark'|'system'} mode
   */
  static update(mode) {
    const newTheme = mode === this.Mode.SYSTEM ? this.#systemTheme : mode;

    if (newTheme !== this.resolvedTheme) {
      this.#notify();
    }

    this.#apply(newTheme, { persist: mode !== this.Mode.SYSTEM });

    if (mode === this.Mode.SYSTEM) {
      this.#clearStorage();
    }
  }
}

Theme.init();

export default Theme;
