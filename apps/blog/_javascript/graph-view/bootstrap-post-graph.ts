import { createGraphView } from './graph-view';

const el = document.getElementById('post-graph-root');
if (el) {
  const dataUrl = el.dataset.graphUrl;
  if (!dataUrl) {
    el.textContent = '그래프 데이터 URL이 없습니다.';
  } else {
    createGraphView({
      container: el,
      dataUrl,
      onNodeOpen(node) {
        if (node.href) window.location.href = node.href;
      }
    }).catch((err) => {
      console.error(err);
      el.textContent = '그래프를 불러오지 못했습니다.';
    });
  }
}
