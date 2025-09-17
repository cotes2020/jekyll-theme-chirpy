/**
 * This script make #search-result-wrapper switch to unload or shown automatically.
 */

const btnSbTrigger = document.getElementById('sidebar-trigger');
const btnSearchTrigger = document.getElementById('search-trigger');
const btnCancel = document.getElementById('search-cancel');
const content = document.querySelectorAll('#main-wrapper>.container>.row');
const topbarTitle = document.getElementById('topbar-title');
const search = document.getElementById('search');
const resultWrapper = document.getElementById('search-result-wrapper');
const results = document.getElementById('search-results');
const input = document.getElementById('search-input');
const hints = document.getElementById('search-hints');

// CSS class names
const LOADED = 'd-block';
const UNLOADED = 'd-none';
const FOCUS = 'input-focus';
const FLEX = 'd-flex';

/* Actions in mobile screens (Sidebar hidden) */
class MobileSearchBar {
  static on() {
    btnSbTrigger.classList.add(UNLOADED);
    topbarTitle.classList.add(UNLOADED);
    btnSearchTrigger.classList.add(UNLOADED);
    search.classList.add(FLEX);
    btnCancel.classList.add(LOADED);
  }

  static off() {
    btnCancel.classList.remove(LOADED);
    search.classList.remove(FLEX);
    btnSbTrigger.classList.remove(UNLOADED);
    topbarTitle.classList.remove(UNLOADED);
    btnSearchTrigger.classList.remove(UNLOADED);
  }
}

class ResultSwitch {
  static resultVisible = false;

  static on() {
    if (!this.resultVisible) {
      resultWrapper.classList.remove(UNLOADED);
      content.forEach((el) => {
        el.classList.add(UNLOADED);
      });
      this.resultVisible = true;
    }
  }

  static off() {
    if (this.resultVisible) {
      results.innerHTML = '';

      if (hints.classList.contains(UNLOADED)) {
        hints.classList.remove(UNLOADED);
      }

      resultWrapper.classList.add(UNLOADED);
      content.forEach((el) => {
        el.classList.remove(UNLOADED);
      });
      input.textContent = '';
      this.resultVisible = false;
    }
  }
}

function isMobileView() {
  return btnCancel.classList.contains(LOADED);
}

export function displaySearch() {
  btnSearchTrigger.addEventListener('click', () => {
    MobileSearchBar.on();
    ResultSwitch.on();
    input.focus();
  });

  btnCancel.addEventListener('click', () => {
    MobileSearchBar.off();
    ResultSwitch.off();
  });

  input.addEventListener('focus', () => {
    search.classList.add(FOCUS);
  });

  input.addEventListener('focusout', () => {
    search.classList.remove(FOCUS);
  });

  input.addEventListener('input', () => {
    if (input.value === '') {
      if (isMobileView()) {
        hints.classList.remove(UNLOADED);
      } else {
        ResultSwitch.off();
      }
    } else {
      ResultSwitch.on();
      if (isMobileView()) {
        hints.classList.add(UNLOADED);
      }
    }
  });
}
