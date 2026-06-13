/**
 * Mermaid-js loader
 */

const MERMAID = 'mermaid';
const themeMapper = Theme.getThemeMapper('default', 'dark');

function refreshTheme(event) {
  if (event.source === window && event.data && event.data.id === Theme.ID) {
    // Re-render the SVG › <https://github.com/mermaid-js/mermaid/issues/311#issuecomment-332557344>
    const mermaidList = document.getElementsByClassName(MERMAID);

    [...mermaidList].forEach((elem) => {
      const svgCode = elem.previousSibling.children.item(0).textContent;
      elem.textContent = svgCode;
      elem.removeAttribute('data-processed');
    });

    const newTheme = themeMapper[Theme.visualState];

    mermaid.initialize({ theme: newTheme });
    mermaid.init(null, `.${MERMAID}`);
  }
}

function setNode(elem) {
  const svgCode = elem.textContent;
  const backup = elem.parentElement;
  backup.classList.add('d-none');
  // Create mermaid node
  const mermaid = document.createElement('pre');
  mermaid.classList.add(MERMAID);
  const text = document.createTextNode(svgCode);
  mermaid.appendChild(text);
  backup.after(mermaid);
}

export function loadMermaid() {
  if (
    typeof mermaid === 'undefined' ||
    typeof mermaid.initialize !== 'function'
  ) {
    return;
  }

  const initTheme = themeMapper[Theme.visualState];

  let mermaidConf = {
    theme: initTheme,
    startOnLoad: false
  };

  const basicList = document.getElementsByClassName('language-mermaid');
  [...basicList].forEach(setNode);

  mermaid.initialize(mermaidConf);

  // Defer rendering until web fonts are ready so Mermaid measures label
  // widths with the correct font metrics, preventing text clipping inside
  // SVG foreignObject when a web font is wider than the fallback font.
  // Cap the wait at 2.5 s so a slow/blocked font source never stalls diagrams.
  const fontsReady = document.fonts?.ready ?? Promise.resolve();
  const timeout = new Promise((resolve) => setTimeout(resolve, 2500));
  Promise.race([fontsReady, timeout]).then(() =>
    mermaid.run().catch(console.error)
  );

  if (Theme.switchable) {
    window.addEventListener('message', refreshTheme);
  }
}
