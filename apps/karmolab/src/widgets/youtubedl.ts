(function (): void {
  'use strict';

  type VideoInfoOk = { videoId: string; title: string; thumbnail: string; url: string };
  type VideoInfoErr = { error: string };
  type VideoInfo = VideoInfoOk | VideoInfoErr;

  function getVideoId(url: string | undefined): string | null {
    if (!url || typeof url !== 'string') return null;
    const trimmed = url.trim();
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
      /^([a-zA-Z0-9_-]{11})$/
    ];
    for (const re of patterns) {
      const m = trimmed.match(re);
      if (m) return m[1] ?? null;
    }
    return null;
  }

  async function fetchVideoInfo(url: string): Promise<VideoInfo> {
    const videoId = getVideoId(url);
    if (!videoId) return { error: '유효한 YouTube URL을 입력해주세요.' };

    const thumbnail = `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
    let title = '';

    try {
      const noembedUrl = `https://noembed.com/embed?url=${encodeURIComponent(url.startsWith('http') ? url : `https://www.youtube.com/watch?v=${videoId}`)}`;
      const res = await fetch(noembedUrl);
      if (res.ok) {
        const data = (await res.json()) as { title?: string };
        title = data.title ?? '';
      }
    } catch {
      title = '(제목을 가져올 수 없음)';
    }

    return { videoId, title: title || '(제목 없음)', thumbnail, url: url.trim() };
  }

  function build(container: HTMLElement): void {
    let currentInfo: VideoInfoOk | null = null;

    const urlGroup = document.createElement('div');
    urlGroup.className = 'field-group';
    const urlLabel = document.createElement('label');
    urlLabel.className = 'field-label';
    urlLabel.htmlFor = 'ytdlUrl';
    urlLabel.textContent = 'YouTube URL';
    const urlInput = document.createElement('input');
    urlInput.type = 'url';
    urlInput.id = 'ytdlUrl';
    urlInput.placeholder = 'https://www.youtube.com/watch?v=...';
    urlGroup.appendChild(urlLabel);
    urlGroup.appendChild(urlInput);
    container.appendChild(urlGroup);
    urlInput.value = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ';

    const btnRow = document.createElement('div');
    btnRow.className = 'field-group';
    btnRow.style.marginTop = '8px';
    const btnFetch = document.createElement('button');
    btnFetch.className = 'btn btn-primary';
    btnFetch.textContent = '영상 정보 가져오기';
    btnRow.appendChild(btnFetch);
    container.appendChild(btnRow);

    const infoCard = document.createElement('div');
    infoCard.id = 'ytdlInfoCard';
    infoCard.className = 'ytdl-info-card';
    infoCard.style.display = 'none';
    infoCard.innerHTML = `
            <div class="ytdl-info-inner">
                <img id="ytdlThumb" alt="" class="ytdl-thumb">
                <div class="ytdl-meta">
                    <h3 id="ytdlTitle" class="ytdl-title"></h3>
                    <div class="ytdl-actions">
                        <button type="button" class="btn btn-primary" id="ytdlMp3">MP3 다운로드</button>
                        <button type="button" class="btn btn-primary" id="ytdlMp4">MP4 다운로드</button>
                    </div>
                    <p id="ytdlStatus" class="ytdl-status"></p>
                </div>
            </div>
        `;
    container.appendChild(infoCard);

    const apiGroup = document.createElement('div');
    apiGroup.className = 'field-group';
    apiGroup.style.marginTop = '12px';
    apiGroup.innerHTML = `
            <label class="field-label" for="ytdlApiBase">yt-api 서버 URL (필수)</label>
            <input type="url" id="ytdlApiBase" class="mono-input" placeholder="http://141.164.45.135:5000">
        `;
    container.appendChild(apiGroup);

    const apiInput = apiGroup.querySelector('#ytdlApiBase') as HTMLInputElement | null;
    if (apiInput && Toolbox.getPref) apiInput.value = Toolbox.getPref('ytdl_api_base', 'http://141.164.45.135:5000') || '';
    if (apiInput && Toolbox.setPref) {
      apiInput.onblur = function (): void {
        Toolbox.setPref?.('ytdl_api_base', apiInput.value.trim());
      };
    }

    Mdd.injectCSS(
      'youtubedl',
      `
            .ytdl-info-card { margin-top:16px; border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; background:var(--bg-tertiary); }
            .ytdl-info-inner { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
            .ytdl-thumb { width:320px; max-width:100%; border-radius:var(--radius-md); display:block; }
            .ytdl-meta { flex:1; min-width:200px; }
            .ytdl-title { font-size:1rem; margin:0 0 12px 0; font-weight:600; line-height:1.4; }
            .ytdl-actions { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
            .ytdl-status { font-size:var(--font-size-xs); color:var(--text-secondary); margin:0; }
            .ytdl-doc { margin-top:20px; padding:14px; border:1px solid var(--border); border-radius:var(--radius-md); background:var(--bg-tertiary); font-size:var(--font-size-xs); line-height:1.6; color:var(--text-secondary); }
            .ytdl-doc h4 { margin:0 0 8px 0; font-size:var(--font-size-sm); color:var(--text-primary); }
            .ytdl-doc p { margin:0 0 8px 0; }
            .ytdl-doc p:last-child { margin-bottom:0; }
        `
    );

    const docEl = document.createElement('div');
    docEl.className = 'ytdl-doc';
    docEl.innerHTML = `
            <h4>사용 방법</h4>
            <p>1. <strong>yt-api 서버 URL</strong> 입력 (예: http://141.164.45.135:5000)</p>
            <p>2. YouTube URL 입력 후 <strong>영상 정보 가져오기</strong> 클릭</p>
            <p>3. <strong>MP3</strong> 또는 <strong>MP4</strong> 다운로드 클릭 → 우리 서버 경유로 다운로드</p>
            <p><strong>참고</strong> 다운로드는 서버 트래픽을 사용합니다. yt-api가 배포된 서버에서 실행 중이어야 합니다.</p>
        `;
    container.appendChild(docEl);

    const thumbEl = infoCard.querySelector('#ytdlThumb') as HTMLImageElement | null;
    const titleEl = infoCard.querySelector('#ytdlTitle') as HTMLElement | null;
    const statusEl = infoCard.querySelector('#ytdlStatus') as HTMLElement | null;
    const btnMp3 = infoCard.querySelector('#ytdlMp3') as HTMLButtonElement | null;
    const btnMp4 = infoCard.querySelector('#ytdlMp4') as HTMLButtonElement | null;

    function getApiBase(): string {
      const base = apiInput?.value?.trim() || (Toolbox.getPref && Toolbox.getPref('ytdl_api_base', '')) || '';
      if (base && Toolbox.setPref) Toolbox.setPref('ytdl_api_base', base);
      return base;
    }

    btnFetch.onclick = async function (): Promise<void> {
      const url = urlInput?.value?.trim();
      if (!url) {
        Toolbox.showToast?.('URL을 입력해주세요.', 'error', undefined);
        return;
      }
      if (!getVideoId(url)) {
        Toolbox.showToast?.('올바른 YouTube URL이 아닙니다.', 'error', undefined);
        return;
      }
      btnFetch.disabled = true;
      try {
        const info = await fetchVideoInfo(url);
        if ('error' in info) {
          Toolbox.showToast?.(info.error, 'error', undefined);
          return;
        }
        currentInfo = info;
        infoCard.style.display = 'block';
        if (thumbEl) {
          thumbEl.src = info.thumbnail;
          thumbEl.alt = info.title;
        }
        if (titleEl) titleEl.textContent = info.title;
        if (statusEl) statusEl.textContent = '';
        Toolbox.showToast?.('영상 정보를 불러왔습니다.', undefined, undefined);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : '정보 조회 실패';
        Toolbox.showToast?.(msg, 'error', undefined);
      } finally {
        btnFetch.disabled = false;
      }
    };

    async function doDownload(fmt: 'mp3' | 'mp4'): Promise<void> {
      if (!currentInfo?.url) return;
      const base = getApiBase();
      if (!base) {
        Toolbox.showToast?.('yt-api 서버 URL을 입력해주세요.', 'error', undefined);
        return;
      }
      const streamUrl =
        base.replace(/\/$/, '') + '/api/yt/stream?url=' + encodeURIComponent(currentInfo.url) + '&format=' + fmt;
      const btn = fmt === 'mp3' ? btnMp3 : btnMp4;
      if (btn) btn.disabled = true;
      if (statusEl) statusEl.textContent = '다운로드 요청 중... (서버에서 처리 중이면 시간이 걸릴 수 있습니다)';
      window.open(streamUrl, '_blank');
      if (statusEl) statusEl.textContent = '다운로드 요청을 보냈습니다. 새 탭을 확인하세요.';
      Toolbox.showToast?.(`${fmt.toUpperCase()} 다운로드 요청`, undefined, undefined);
      if (btn) btn.disabled = false;
    }

    if (btnMp3) btnMp3.onclick = () => void doDownload('mp3');
    if (btnMp4) btnMp4.onclick = () => void doDownload('mp4');
  }

  Toolbox.register({
    id: 'ytdownloader',
    title: '유튜브 다운로드',
    desc: '유튜브 영상을 다운로드합니다',
    layout: 'form',
    icon: '<path d="M10 16.5l6-4.5-6-4.5v9zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>',
    tabs: [{ id: 'main', label: '다운로드', build }]
  });
})();
