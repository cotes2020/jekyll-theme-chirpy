/**
 * This script make #search-result-wrapper switch to unloaded or shown automatically.
 */
const $btnSbTrigger = $('#sidebar-trigger');
const $btnSearchTrigger = $('#search-trigger');
const $btnCancel = $('#search-cancel');
const $content = $('#main>.row');
const $topbarTitle = $('#topbar-title');
const $searchWrapper = $('#search-wrapper');
const $resultWrapper = $('#search-result-wrapper');
const $results = $('#search-results');
const $input = $('#search-input');
const $hints = $('#search-hints');
const $viewport = $('html,body');

// class names
const C_LOADED = 'loaded';
const C_UNLOADED = 'unloaded';
const C_FOCUS = 'input-focus';
const C_FLEX = 'd-flex';

class ScrollBlocker {
  static offset = 0;
  static resultVisible = false;

  static on() {
    ScrollBlocker.offset = window.scrollY;
    $viewport.scrollTop(0);
  }

  static off() {
    $viewport.scrollTop(ScrollBlocker.offset);
  }
}

/*--- Actions in mobile screens (Sidebar hidden) ---*/
class MobileSearchBar {
  static on() {
    $btnSbTrigger.addClass(C_UNLOADED);
    $topbarTitle.addClass(C_UNLOADED);
    $btnSearchTrigger.addClass(C_UNLOADED);
    $searchWrapper.addClass(C_FLEX);
    $btnCancel.addClass(C_LOADED);
  }

  static off() {
    $btnCancel.removeClass(C_LOADED);
    $searchWrapper.removeClass(C_FLEX);
    $btnSbTrigger.removeClass(C_UNLOADED);
    $topbarTitle.removeClass(C_UNLOADED);
    $btnSearchTrigger.removeClass(C_UNLOADED);
  }
}

class ResultSwitch {
  static on() {
    if (!ScrollBlocker.resultVisible) {
      // the block method must be called before $(#main) unloaded.
      ScrollBlocker.on();
      $resultWrapper.removeClass(C_UNLOADED);
      $content.addClass(C_UNLOADED);
      ScrollBlocker.resultVisible = true;
    }
  }

  static off() {
    if (ScrollBlocker.resultVisible) {
      $results.empty();
      if ($hints.hasClass(C_UNLOADED)) {
        $hints.removeClass(C_UNLOADED);
      }
      $resultWrapper.addClass(C_UNLOADED);
      $content.removeClass(C_UNLOADED);

      // now the release method must be called after $(#main) display
      ScrollBlocker.off();

      $input.val('');
      ScrollBlocker.resultVisible = false;
    }
  }
}

function isMobileView() {
  return $btnCancel.hasClass(C_LOADED);
}

export function displaySearch() {
  $btnSearchTrigger.on('click', function () {
    MobileSearchBar.on();
    ResultSwitch.on();
    $input.trigger('focus');
  });

  $btnCancel.on('click', function () {
    MobileSearchBar.off();
    ResultSwitch.off();
  });

  $input.on('focus', function () {
    $searchWrapper.addClass(C_FOCUS);
  });

  $input.on('focusout', function () {
    $searchWrapper.removeClass(C_FOCUS);
  });

  $input.on('input', () => {
    if ($input.val() === '') {
      if (isMobileView()) {
        $hints.removeClass(C_UNLOADED);
      } else {
        ResultSwitch.off();
      }
    } else {
      ResultSwitch.on();
      if (isMobileView()) {
        $hints.addClass(C_UNLOADED);
      }
    }
  });
}
