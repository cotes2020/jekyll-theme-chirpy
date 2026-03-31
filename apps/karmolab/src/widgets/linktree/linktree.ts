/**
 * Link Tree — 링크 인 바이오 스타일 페이지
 * 프로필 + 링크 카드 목록
 */
(function (): void {
  const ICONS: Record<string, string> = {
    email:
      '<path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>',
    github:
      '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>',
    twitter:
      '<path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-5.8 1.1 0 3-1.2 3-1.2z"/>',
    linkedin:
      '<path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/>',
    blog:
      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
    link:
      '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    copy:
      '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
  };

  type LinkItem = {
    url?: string;
    copyTarget?: string;
    label: string;
    desc?: string;
    icon?: string;
  };

  type LinkGroup = { group: string; items: LinkItem[] };

  const CONFIG: {
    avatar: string;
    name: string;
    bio: string;
    groups: LinkGroup[];
  } = {
    avatar: '/assets/img/profile/star-transparent.png',
    name: 'mascari4615',
    bio: '삶을 섞고 술을 바꿀 시간',
    groups: [
      {
        group: '연락처',
        items: [
          { copyTarget: 'Mascari4615@gmail.com', label: '이메일', desc: 'Mascari4615@gmail.com', icon: 'email' },
          { url: 'https://github.com/mascari4615', label: 'GitHub', desc: '코드 & 프로젝트', icon: 'github' },
          { url: 'https://twitter.com/mascari4615', label: 'Twitter / X', desc: '짧은 생각', icon: 'twitter' },
          { url: 'https://www.linkedin.com/in/도윤-김-b89049194/', label: 'LinkedIn', desc: '커리어', icon: 'linkedin' }
        ]
      },
      {
        group: '블로그·프로젝트',
        items: [{ url: 'https://mascari4615.github.io/', label: '블로그', desc: '메모, 기록, 전략', icon: 'blog' }]
      },
      { group: '기타', items: [] }
    ]
  };

  Mdd.injectCSS(
    'linktree',
    `
        .linktree-wrap { display: flex; flex-direction: column; align-items: center; padding: var(--space-xl) 0; max-width: 400px; margin: 0 auto; }
        .linktree-intro { font-size: var(--font-size-sm); font-weight: 600; color: var(--text-secondary); margin-bottom: var(--space-md); letter-spacing: -0.02em; display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap; }
        .linktree-intro-btn { background: none; border: none; font: inherit; color: inherit; cursor: pointer; padding: 0; text-decoration: underline; text-underline-offset: 2px; }
        .linktree-intro-btn:hover { color: var(--accent); }
        .linktree-avatar { width: 88px; height: 88px; border-radius: 50%; object-fit: cover; border: 3px solid var(--border); margin-bottom: var(--space-md); }
        .linktree-name { font-size: var(--font-size-lg); font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; margin-bottom: 4px; }
        .linktree-bio { font-size: var(--font-size-xs); color: var(--text-tertiary); margin-bottom: var(--space-lg); text-align: center; line-height: 1.5; }
        .linktree-groups { display: flex; flex-direction: column; gap: var(--space-lg); width: 100%; }
        .linktree-group { display: flex; flex-direction: column; gap: var(--space-sm); }
        .linktree-group-title { font-size: var(--font-size-2xs); font-weight: 600; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
        .linktree-list { display: flex; flex-direction: column; gap: var(--space-sm); }
        .linktree-card { display: flex; align-items: center; gap: 12px; padding: 14px 18px; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: var(--radius-lg); color: var(--text-primary); text-decoration: none; font-weight: 500; font-size: var(--font-size-sm); transition: all var(--transition); text-align: left; }
        .linktree-card:hover { background: var(--bg-hover); border-color: var(--accent); color: var(--accent); transform: translateY(-1px); }
        .linktree-card-email { cursor: default; }
        .linktree-card-icon { flex-shrink: 0; width: 20px; height: 20px; opacity: 0.7; }
        .linktree-card-icon svg { width: 100%; height: 100%; }
        .linktree-card:hover .linktree-card-icon { opacity: 1; }
        .linktree-card-body { flex: 1; min-width: 0; }
        .linktree-card-desc { font-size: var(--font-size-2xs); font-weight: 400; color: var(--text-tertiary); margin-top: 2px; }
        .linktree-card:hover .linktree-card-desc { color: var(--text-secondary); }
        .linktree-card-copy { flex-shrink: 0; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 6px; color: var(--text-tertiary); background: none; border: none; cursor: pointer; opacity: 0.6; transition: all var(--transition); }
        .linktree-card-copy:hover { opacity: 1; color: var(--accent); }
        .linktree-card-copy svg { width: 16px; height: 16px; }
    `
  );

  function build(container: HTMLElement): void {
    Mdd.linePreset('home_hub', { msg: '링크 모아뒀어요~' });

    const wrap = document.createElement('div');
    wrap.className = 'linktree-wrap';

    const linktreeUrl = location.origin + location.pathname + '#linktree';
    const email = 'Mascari4615@gmail.com';
    function copyAndToast(text: string, btn: HTMLButtonElement): void {
      function done(): void {
        Toolbox.showToast?.('클립보드에 복사됨', undefined, undefined);
        const t = btn.title;
        btn.title = '복사됨!';
        setTimeout(function () {
          btn.title = t;
        }, 2000);
      }
      if (navigator.clipboard?.writeText) {
        void navigator.clipboard.writeText(text).then(done);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        done();
      }
    }
    const introEl = document.createElement('div');
    introEl.className = 'linktree-intro';
    const emailBtn = document.createElement('button');
    emailBtn.type = 'button';
    emailBtn.className = 'linktree-intro-btn';
    emailBtn.textContent = '이메일';
    emailBtn.title = '클립보드에 복사';
    emailBtn.onclick = function (): void {
      copyAndToast(email, emailBtn);
    };
    const linkBtn = document.createElement('button');
    linkBtn.type = 'button';
    linkBtn.className = 'linktree-intro-btn';
    linkBtn.textContent = '링크';
    linkBtn.title = '클립보드에 복사';
    linkBtn.onclick = function (): void {
      copyAndToast(linktreeUrl, linkBtn);
    };
    introEl.appendChild(emailBtn);
    introEl.appendChild(document.createTextNode(' · '));
    introEl.appendChild(linkBtn);
    wrap.appendChild(introEl);

    const avatar = document.createElement('img');
    avatar.className = 'linktree-avatar';
    avatar.src = CONFIG.avatar;
    avatar.alt = CONFIG.name;
    avatar.onerror = (): void => {
      avatar.style.display = 'none';
    };
    wrap.appendChild(avatar);

    const nameEl = document.createElement('h1');
    nameEl.className = 'linktree-name';
    nameEl.textContent = CONFIG.name;
    wrap.appendChild(nameEl);

    const bioEl = document.createElement('p');
    bioEl.className = 'linktree-bio';
    bioEl.textContent = CONFIG.bio;
    wrap.appendChild(bioEl);

    const groupsWrap = document.createElement('div');
    groupsWrap.className = 'linktree-groups';
    CONFIG.groups.forEach(function (g) {
      if (!g.items || g.items.length === 0) return;
      const groupDiv = document.createElement('div');
      groupDiv.className = 'linktree-group';
      const title = document.createElement('div');
      title.className = 'linktree-group-title';
      title.textContent = g.group;
      groupDiv.appendChild(title);
      const list = document.createElement('div');
      list.className = 'linktree-list';
      (g.items || []).forEach(function (link: LinkItem) {
        const isEmail = !!link.copyTarget;
        const card = document.createElement(isEmail ? 'div' : 'a');
        card.className = 'linktree-card' + (isEmail ? ' linktree-card-email' : '');
        if (!isEmail && link.url) {
          (card as HTMLAnchorElement).href = link.url;
          (card as HTMLAnchorElement).target = '_blank';
          (card as HTMLAnchorElement).rel = 'noopener noreferrer';
        }
        card.title = link.label;
        const iconKey = link.icon || 'link';
        const iconPath = ICONS[iconKey] ?? ICONS.link;
        const body = link.desc
          ? '<span class="linktree-card-body">' +
            link.label +
            '<div class="linktree-card-desc">' +
            link.desc +
            '</div></span>'
          : '<span class="linktree-card-body">' + link.label + '</span>';
        let html =
          '<span class="linktree-card-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
          iconPath +
          '</svg></span>' +
          body;
        if (isEmail) {
          html +=
            '<button type="button" class="linktree-card-copy" title="클립보드에 복사"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
            ICONS.copy +
            '</svg></button>';
        }
        card.innerHTML = html;
        if (isEmail && link.copyTarget) {
          const copyBtn = card.querySelector('.linktree-card-copy');
          const text = link.copyTarget;
          if (copyBtn instanceof HTMLButtonElement) {
            const btn = copyBtn;
            btn.addEventListener('click', function (e: Event) {
              e.stopPropagation();
              function done(): void {
                Toolbox.showToast?.('클립보드에 복사됨', undefined, undefined);
                btn.title = '복사됨!';
                setTimeout(function () {
                  btn.title = '클립보드에 복사';
                }, 2000);
              }
              if (navigator.clipboard?.writeText) {
                void navigator.clipboard.writeText(text).then(done);
              } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                done();
              }
            });
          }
        }
        list.appendChild(card);
      });
      groupDiv.appendChild(list);
      groupsWrap.appendChild(groupDiv);
    });
    wrap.appendChild(groupsWrap);

    container.innerHTML = '';
    container.appendChild(wrap);
  }

  Toolbox.register({
    id: 'linktree',
    title: '링크',
    desc: '개발자 연락처 & 링크 모음',
    layout: 'narrow',
    icon: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    tabs: [{ id: 'linktree-main', label: '링크', build }]
  });
})();
