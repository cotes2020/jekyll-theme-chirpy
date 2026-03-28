import { createGraphView } from './graph-view.js';

const el = document.getElementById('post-graph-root');
if (el) {
  const dataUrl = el.dataset.graphUrl;
  if (!dataUrl) {
    el.textContent = '??? ??? URL? ????.';
  } else {
    createGraphView({
      container: el,
      dataUrl,
      onNodeOpen(node) {
        if (node.href) window.location.href = node.href;
      }
    }).catch((err) => {
      console.error(err);
      el.textContent = '???? ???? ?????.';
    });
  }
}
