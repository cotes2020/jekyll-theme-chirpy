/**
 * Mermaid-js loader
 */

import { mermaid } from '../globals-mermaid';
import { Theme } from '../globals-theme';

const MERMAID = 'mermaid';
const themeMapper = Theme.getThemeMapper('default', 'dark');

function refreshTheme(event: MessageEvent): void {
  if (event.source === window && event.data && event.data.id === Theme.ID) {
    // Re-render the SVG › <https://github.com/mermaid-js/mermaid/issues/311#issuecomment-332557344>
    const mermaidList = document.getElementsByClassName(MERMAID);

    [...mermaidList].forEach((elem) => {
      const prev = elem.previousSibling;
      if (!(prev instanceof HTMLElement)) return;
      const svgCode = prev.children.item(0)?.textContent ?? '';
      elem.textContent = svgCode;
      elem.removeAttribute('data-processed');
    });

    const newTheme = themeMapper[Theme.visualState];

    mermaid.initialize({ theme: newTheme });
    mermaid.init(null, `.${MERMAID}`);
  }
}

function setNode(elem: Element): void {
  const svgCode = elem.textContent ?? '';
  const backup = elem.parentElement;
  if (!backup) return;
  backup.classList.add('d-none');
  // Create mermaid node
  const mermaidNode = document.createElement('pre');
  mermaidNode.classList.add(MERMAID);
  const text = document.createTextNode(svgCode);
  mermaidNode.appendChild(text);
  backup.after(mermaidNode);
}

export function loadMermaid(): void {
  const globalMermaid = (window as Window & { mermaid?: unknown }).mermaid;
  if (typeof globalMermaid === 'undefined') {
    return;
  }

  const initTheme = themeMapper[Theme.visualState];

  const mermaidConf = {
    theme: initTheme
  };

  const basicList = document.getElementsByClassName('language-mermaid');
  [...basicList].forEach(setNode);

  mermaid.initialize(mermaidConf);

  if (Theme.switchable) {
    window.addEventListener('message', refreshTheme);
  }
}
