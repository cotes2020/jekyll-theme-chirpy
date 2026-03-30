/**
 * imageconvert 로더:
 * core.js -> batch-pipeline.js -> widget.js 순서로 동적 로드하고,
 * 결과 Promise를 `window.KARMOLAB_WIDGET_LOADER_WAIT`에 등록합니다.
 */
(function (): void {
  function baseUrl(): string {
    const s = document.currentScript as HTMLScriptElement | null;
    if (s?.src) {
      try {
        const u = new URL(s.src);
        return u.origin + u.pathname.replace(/\/[^/]+$/, '/');
      } catch (_) {}
    }
    return (location.origin || '') + '/apps/karmolab/js/widgets/imageconvert/';
  }

  function loadSeq(urls: string[]): Promise<void> {
    return urls.reduce<Promise<void>>((p, url) => {
      return p.then(() => {
        return new Promise<void>((res, rej) => {
          const el = document.createElement('script');
          el.src = url;
          el.onload = () => res();
          el.onerror = () => rej(new Error('failed to load: ' + url));
          document.body.appendChild(el);
        });
      });
    }, Promise.resolve());
  }

  const base = baseUrl();
  const p = loadSeq([base + 'core.js', base + 'batch-pipeline.js', base + 'widget.js']);

  try {
    window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
    window.KARMOLAB_WIDGET_LOADER_WAIT.push(p);
  } catch (_) {}

  p.catch((err: unknown) => {
    try {
      Toolbox.showToast?.('이미지 변환 로드 실패', 'error', err);
    } catch (_) {}
    console.error(err);
  });
})();
