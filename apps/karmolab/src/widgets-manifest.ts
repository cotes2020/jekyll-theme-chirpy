/**
 * 위젯 매니페스트
 *
 * - KARMOLAB_WIDGETS_BOOT: 초기 로드(즉시 실행)
 * - 지연 위젯 메타·경로: widgets-lazy-meta.js 의 KARMOLAB_LAZY_META (단일 출처)
 *
 * 위젯 추가: boot에 넣거나 lazy-meta에 항목 추가 후 해당 위젯은
 * (디버그: widgets/devtools.js — Tauri는 동일 파일을 앱에 포함해 init 전 주입)
 *   Toolbox.register({ ...Toolbox.getLazyWidgetPublicMeta('id'), tabs: [...] })
 */
window.KARMOLAB_WIDGETS_BOOT = [
  'favorites',
  'linktree/linktree',
  'user',
  'dashboard',
  'docs/docs',
  'servermonitor',
  'devtools',
  'randomgen/randomgen'
];
