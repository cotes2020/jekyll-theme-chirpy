;(function (): void {
  // tierlist 위젯 엔트리포인트(매니페스트용 단일 파일)
  // 이 파일은 widgets/tierlist/ 폴더 안에 있으므로, baseUrl은 "현재 파일의 폴더"면 충분합니다.

  function baseUrl(): string {
    const scripts = document.scripts
    const current =
      (document.currentScript as HTMLScriptElement | null) ??
      (scripts.length ? (scripts[scripts.length - 1] as HTMLScriptElement) : null)

    if (current?.src) {
      try {
        const u = new URL(current.src)
        // .../widgets/tierlist/tierlist.js → .../widgets/tierlist/
        return u.origin + u.pathname.replace(/\/[^/]+$/, '/')
      } catch {
        // ignore and fall back
      }
    }
    return (location.origin || '') + '/apps/karmolab/js/widgets/tierlist/'
  }

  function loadSeq(urls: string[]): Promise<void> {
    return urls.reduce<Promise<void>>(
      (p, url) =>
        p.then(
          () =>
            new Promise<void>((res, rej) => {
              const s = document.createElement('script')
              s.src = url
              s.onload = () => res()
              s.onerror = () => rej(new Error('failed to load: ' + url))
              document.body.appendChild(s)
            })
        ),
      Promise.resolve()
    )
  }

  const base = baseUrl()
  const files = [
    'namespace.js',
    'styles.js',
    'storage.js',
    'ui.js',
    'dnd.js',
    'publish.js',
    'dialogs.js',
    'render.js',
    'index.js'
  ].map((f) => base + f)

  const p = loadSeq(files)

  // widgets-loader가 init 전에 기다릴 수 있도록 Promise를 공유
  try {
    window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || []
    window.KARMOLAB_WIDGET_LOADER_WAIT.push(p)
  } catch {
    // ignore
  }

  p.catch((err: unknown) => {
    try {
      Toolbox.showToast?.('tierlist 로드 실패', 'error', err)
    } catch {
      // ignore toast failure
    }
    // eslint-disable-next-line no-console
    console.error(err)
  })
})()

