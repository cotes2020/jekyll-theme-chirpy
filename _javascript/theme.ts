/**
 * Theme management class
 *
 * To reduce flickering during page load, this script should be loaded synchronously.
 */
class Theme {
  static #modeKey = 'mode';
  static #modeAttr = 'data-mode';
  static #darkMedia = window.matchMedia('(prefers-color-scheme: dark)');
  static switchable = !document.documentElement.hasAttribute(this.#modeAttr);

  static get DARK(): string {
    return 'dark';
  }

  static get LIGHT(): string {
    return 'light';
  }

  /**
   * @returns {string} Theme mode identifier
   */
  static get ID(): string {
    return 'theme-mode';
  }

  /**
   * Gets the current visual state of the theme.
   *
   * @returns {string} The current visual state, either the mode if it exists,
   *                   or the system dark mode state ('dark' or 'light').
   */
  static get visualState(): string {
    if (this.#hasMode) {
      return this.#mode as string;
    } else {
      return this.#sysDark ? this.DARK : this.LIGHT;
    }
  }

  static get #mode(): string | null {
    return (
      sessionStorage.getItem(this.#modeKey) ||
      document.documentElement.getAttribute(this.#modeAttr)
    );
  }

  static get #isDarkMode(): boolean {
    return this.#mode === this.DARK;
  }

  static get #hasMode(): boolean {
    return this.#mode !== null;
  }

  static get #sysDark(): boolean {
    return this.#darkMedia.matches;
  }

  /**
   * Maps theme modes to provided values
   * @param {string} light Value for light mode
   * @param {string} dark Value for dark mode
   * @returns {Object} Mapped values
   */
  static getThemeMapper(light: string, dark: string): Record<string, string> {
    return {
      [this.LIGHT]: light,
      [this.DARK]: dark
    };
  }

  /**
   * Initializes the theme based on system preferences or stored mode
   */
  static init(): void {
    if (!this.switchable) {
      return;
    }

    this.#darkMedia.addEventListener('change', () => {
      const lastMode = this.#mode;
      this.#clearMode();

      if (lastMode !== this.visualState) {
        this.#notify();
      }
    });

    if (!this.#hasMode) {
      return;
    }

    if (this.#isDarkMode) {
      this.#setDark();
    } else {
      this.#setLight();
    }
  }

  /**
   * Flips the current theme mode
   */
  static flip(): void {
    if (this.#hasMode) {
      this.#clearMode();
    } else if (this.#sysDark) {
      this.#setLight();
    } else {
      this.#setDark();
    }
    this.#notify();
  }

  static #setDark(): void {
    document.documentElement.setAttribute(this.#modeAttr, this.DARK);
    sessionStorage.setItem(this.#modeKey, this.DARK);
  }

  static #setLight(): void {
    document.documentElement.setAttribute(this.#modeAttr, this.LIGHT);
    sessionStorage.setItem(this.#modeKey, this.LIGHT);
  }

  static #clearMode(): void {
    document.documentElement.removeAttribute(this.#modeAttr);
    sessionStorage.removeItem(this.#modeKey);
  }

  /**
   * Notifies other plugins that the theme mode has changed
   */
  static #notify(): void {
    window.postMessage({ id: this.ID }, '*');
  }
}

Theme.init();

export default Theme;
