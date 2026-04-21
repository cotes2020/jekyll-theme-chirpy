---
title: Graph
icon: fas fa-project-diagram
order: 88
---

포스트 본문에 있는 **내부 링크**(`/posts/…`)만 모아 관계를 그립니다. 데이터는 `npm run build:graph`로 `post-graph.json`을 갱신합니다.

UI·상호작용 개념 참고: [Obsidian 그래프 뷰(플러그인 도움말)](https://obsidian.md/ko/help/plugins/graph)

<div
  id="post-graph-root"
  class="post-graph-root"
  data-graph-url="{{ '/assets/js/data/post-graph.json' | relative_url }}"
  style="min-height:70vh;height:70vh;width:100%;border:1px solid var(--main-border-color);border-radius:0.375rem;background:var(--main-bg);"
></div>

<script type="module" src="{{ '/assets/js/graph-view/bootstrap-post-graph.js' | relative_url }}"></script>
