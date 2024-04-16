/**
 * Clipboard functions
 *
 * Dependencies:
 *   - popper.js (https://github.com/popperjs/popper-core)
 *   - clipboard.js (https://github.com/zenorocha/clipboard.js)
 */

const clipboardSelector = '.code-header>button';

const ICON_DEFAULT = 'far fa-clipboard';
const ICON_SUCCESS = 'fas fa-check';

const ATTR_TIMEOUT = 'timeout';
const ATTR_TITLE_SUCCEED = 'data-title-succeed';
const ATTR_TITLE_ORIGIN = 'data-bs-original-title';
const TIMEOUT = 2000; // in milliseconds

function isLocked(node) {
  if (node.hasAttribute(ATTR_TIMEOUT)) {
    let timeout = node.getAttribute(ATTR_TIMEOUT);
    if (Number(timeout) > Date.now()) {
      return true;
    }
  }

  return false;
}

function lock(node) {
  node.setAttribute(ATTR_TIMEOUT, Date.now() + TIMEOUT);
}

function unlock(node) {
  node.removeAttribute(ATTR_TIMEOUT);
}

function showTooltip(btn) {
  const succeedTitle = btn.getAttribute(ATTR_TITLE_SUCCEED);
  btn.setAttribute(ATTR_TITLE_ORIGIN, succeedTitle);
  bootstrap.Tooltip.getInstance(btn).show();
}

function hideTooltip(btn) {
  bootstrap.Tooltip.getInstance(btn).hide();
  btn.removeAttribute(ATTR_TITLE_ORIGIN);
}

function setSuccessIcon(btn) {
  const icon = btn.children[0];
  icon.setAttribute('class', ICON_SUCCESS);
}

function resumeIcon(btn) {
  const icon = btn.children[0];
  icon.setAttribute('class', ICON_DEFAULT);
}

export function initClipboard() {
  const clipboardList = document.querySelectorAll(clipboardSelector);

  if (clipboardList.length === 0) {
    return;
  }

  // Initial the clipboard.js object
  const clipboard = new ClipboardJS(clipboardSelector, {
    target: (trigger) => {
      const codeBlock = trigger.parentNode.nextElementSibling;
      return codeBlock.querySelector('code .rouge-code');
    }
  });

  [...clipboardList].map(
    (elem) =>
      new bootstrap.Tooltip(elem, {
        placement: 'left'
      })
  );

  clipboard.on('success', (e) => {
    const trigger = e.trigger;

    e.clearSelection();

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

  const btnCopyLink = document.getElementById('copy-link');

  btnCopyLink.addEventListener('click', (e) => {
    const target = e.target;

    if (isLocked(target)) {
      return;
    }

    // Copy URL to clipboard
    navigator.clipboard.writeText(window.location.href).then(() => {
      const defaultTitle = target.getAttribute(ATTR_TITLE_ORIGIN);
      const succeedTitle = target.getAttribute(ATTR_TITLE_SUCCEED);

      // Switch tooltip title
      target.setAttribute(ATTR_TITLE_ORIGIN, succeedTitle);
      bootstrap.Tooltip.getInstance(target).show();

      lock(target);

      setTimeout(() => {
        target.setAttribute(ATTR_TITLE_ORIGIN, defaultTitle);
        unlock(target);
      }, TIMEOUT);
    });
  });

  btnCopyLink.addEventListener('mouseleave', (e) => {
    bootstrap.Tooltip.getInstance(e.target).hide();
  });
}
