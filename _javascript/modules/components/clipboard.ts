/**
 * Clipboard functions
 *
 * Dependencies:
 *    clipboard.js (https://github.com/zenorocha/clipboard.js)
 */

import Tooltip from 'bootstrap/js/src/tooltip';
import { ClipboardJS } from '../globals';

const clipboardSelector = '.code-header>button';

const ICON_DEFAULT = 'far fa-clipboard';
const ICON_SUCCESS = 'fas fa-check';

const ATTR_TIMEOUT = 'timeout';
const ATTR_TITLE_SUCCEED = 'data-title-succeed';
const ATTR_TITLE_ORIGIN = 'data-bs-original-title';
const TIMEOUT = 2000; // in milliseconds

function isLocked(node: Element): boolean {
  if (node.hasAttribute(ATTR_TIMEOUT)) {
    const timeout = node.getAttribute(ATTR_TIMEOUT);
    if (Number(timeout) > Date.now()) {
      return true;
    }
  }

  return false;
}

function lock(node: Element): void {
  node.setAttribute(ATTR_TIMEOUT, String(Date.now() + TIMEOUT));
}

function unlock(node: Element): void {
  node.removeAttribute(ATTR_TIMEOUT);
}

function showTooltip(btn: Element): void {
  const succeedTitle = btn.getAttribute(ATTR_TITLE_SUCCEED);
  if (succeedTitle) {
    btn.setAttribute(ATTR_TITLE_ORIGIN, succeedTitle);
  }
  Tooltip.getOrCreateInstance(btn).show();
}

function hideTooltip(btn: Element): void {
  Tooltip.getOrCreateInstance(btn).hide();
  btn.removeAttribute(ATTR_TITLE_ORIGIN);
}

function setSuccessIcon(btn: HTMLElement): void {
  const icon = btn.children.item(0);
  if (!icon) return;
  icon.setAttribute('class', ICON_SUCCESS);
}

function resumeIcon(btn: HTMLElement): void {
  const icon = btn.children.item(0);
  if (!icon) return;
  icon.setAttribute('class', ICON_DEFAULT);
}

function setCodeClipboard(): void {
  const clipboardList = document.querySelectorAll<HTMLElement>(clipboardSelector);

  if (clipboardList.length === 0) {
    return;
  }

  // Initial the clipboard.js object
  const clipboard = new ClipboardJS(clipboardSelector, {
    target: (trigger) => {
      const parent = trigger.parentElement;
      const codeBlock = parent?.nextElementSibling;
      const code = codeBlock?.querySelector<HTMLElement>('code .rouge-code');
      if (!code) {
        throw new Error('Cannot find code element for clipboard copy');
      }
      return code;
    }
  });

  [...clipboardList].map(
    (elem) =>
      new Tooltip(elem, {
        placement: 'left'
      })
  );

  clipboard.on('success', (e) => {
    const trigger = e.trigger as HTMLElement;

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
}

function setLinkClipboard(): void {
  const btnCopyLink = document.getElementById('copy-link');

  if (btnCopyLink === null) {
    return;
  }

  btnCopyLink.addEventListener('click', (e) => {
    const target = e.currentTarget;
    if (!(target instanceof HTMLElement)) return;

    if (isLocked(target)) {
      return;
    }

    // Copy URL to clipboard
    navigator.clipboard.writeText(window.location.href).then(() => {
      const defaultTitle = target.getAttribute(ATTR_TITLE_ORIGIN);
      const succeedTitle = target.getAttribute(ATTR_TITLE_SUCCEED);

      // Switch tooltip title
      if (succeedTitle) {
        target.setAttribute(ATTR_TITLE_ORIGIN, succeedTitle);
      }
      Tooltip.getOrCreateInstance(target).show();

      lock(target);

      setTimeout(() => {
        if (defaultTitle) {
          target.setAttribute(ATTR_TITLE_ORIGIN, defaultTitle);
        } else {
          target.removeAttribute(ATTR_TITLE_ORIGIN);
        }
        unlock(target);
      }, TIMEOUT);
    });
  });

  btnCopyLink.addEventListener('mouseleave', (e) => {
    const target = e.currentTarget;
    if (!(target instanceof HTMLElement)) return;
    Tooltip.getOrCreateInstance(target).hide();
  });
}

export function initClipboard(): void {
  setCodeClipboard();
  setLinkClipboard();
}
