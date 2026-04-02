/**
 * wiki/entities/characters/{slug}.yaml + {slug}.md 를 fetch → 파싱 → KarmoWorld.entities + bindings 채움
 * imagegen/chatbot 스크립트보다 먼저 실행되며, Toolbox 지연 로더가 Promise 완료까지 대기합니다.
 */
(function (): void {
  const SLUGS = ['yon', 'alisa', 'ling'] as const;

  function worldBaseUrl(): string {
    const s = document.currentScript as HTMLScriptElement | null;
    if (s && s.src) {
      try {
        const u = new URL(s.src);
        return u.origin + u.pathname.replace(/\/[^/]+$/, '/');
      } catch (_) {}
    }
    return (location.origin || '') + '/apps/karmolab/world/';
  }

  function wikiUrl(base: string, slug: string): string {
    return base + 'wiki/entities/characters/' + slug + '.md';
  }

  function wikiYamlUrl(base: string, slug: string): string {
    return base + 'wiki/entities/characters/' + slug + '.yaml';
  }

  function str(v: unknown): string {
    return typeof v === 'string' ? v : v == null ? '' : String(v);
  }

  function metaToImagegen(m: Record<string, unknown>) {
    return {
      entityId: str(m.entityId),
      imagegenPresetId: str(m.imagegen_presetId),
      icon: str(m.imagegen_icon) || '📷',
      label: str(m.imagegen_label),
      shortLabel: str(m.imagegen_shortLabel),
      prompt: str(m.imagegen_prompt)
    };
  }

  function metaToChatbot(m: Record<string, unknown>) {
    return {
      entityId: str(m.entityId),
      chatbotId: str(m.chatbot_id),
      name: str(m.chatbot_name),
      userName: str(m.chatbot_userName) || '사용자',
      userNote: str(m.chatbot_userNote),
      visualDescription: str(m.chatbot_visualDescription),
      description: str(m.chatbot_description),
      personality: str(m.chatbot_personality),
      scenario: str(m.chatbot_scenario),
      firstMes: str(m.chatbot_firstMes)
    };
  }

  function metaToEntity(slug: string, m: Record<string, unknown>) {
    const tags: string[] = Array.isArray(m.tags)
      ? m.tags.map((t) => (typeof t === 'string' ? t : String(t)))
      : typeof m.tags === 'string'
        ? [m.tags]
        : [];
    const aliases: string[] = Array.isArray(m.aliases)
      ? m.aliases.map((a) => (typeof a === 'string' ? a : String(a)))
      : [];
    return {
      id: str(m.entityId),
      slug: str(m.slug) || slug,
      nameKo: str(m.nameKo),
      nameEn: str(m.nameEn),
      aliases,
      oneLine: str(m.oneLine),
      tags
    };
  }

  async function loadAll(): Promise<void> {
    const parseSplit = window.KarmoWorld?.parseMd?.parseCharacterWikiFromSplitFiles;
    if (typeof parseSplit !== 'function') {
      throw new Error('KarmoWorld.parseMd.parseCharacterWikiFromSplitFiles 없음');
    }

    const base = worldBaseUrl();
    const imagegen: ReturnType<typeof metaToImagegen>[] = [];
    const chatbot: ReturnType<typeof metaToChatbot>[] = [];
    const characters: Record<string, ReturnType<typeof metaToEntity>> = {};

    for (const slug of SLUGS) {
      const mdUrl = wikiUrl(base, slug);
      const yUrl = wikiYamlUrl(base, slug);
      const [rm, ry] = await Promise.all([fetch(mdUrl), fetch(yUrl)]);
      if (!rm.ok) throw new Error('wiki 로드 실패: ' + mdUrl);
      if (!ry.ok) throw new Error('wiki 로드 실패: ' + yUrl);
      const mdText = await rm.text();
      const yamlText = await ry.text();
      const { meta } = parseSplit(yamlText, mdText);
      if (meta.entityId == null || meta.entityId === '') throw new Error('entityId 없음: ' + slug);
      characters[slug] = metaToEntity(slug, meta);
      imagegen.push(metaToImagegen(meta));
      chatbot.push(metaToChatbot(meta));
    }

    window.KarmoWorld = window.KarmoWorld || {};
    window.KarmoWorld.entities = window.KarmoWorld.entities || {};
    window.KarmoWorld.entities.characters = characters;

    window.KarmoWorld.bindings = window.KarmoWorld.bindings || {};
    window.KarmoWorld.bindings.imagegen = window.KarmoWorld.bindings.imagegen || {};
    window.KarmoWorld.bindings.chatbot = window.KarmoWorld.bindings.chatbot || {};
    window.KarmoWorld.bindings.imagegen.characters = imagegen;
    window.KarmoWorld.bindings.chatbot.characters = chatbot;
  }

  window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
  const p = loadAll().catch((err: unknown) => {
    try {
      console.error('[KarmoWorld] wiki 캐릭터 로드 실패', err);
    } catch (_) {}
    try {
      const toast = typeof Toolbox !== 'undefined' ? Toolbox.showToast : undefined;
      if (toast) toast('세계관 캐릭터 데이터 로드 실패', 'error', undefined);
    } catch (_) {}
  });
  window.KARMOLAB_WIDGET_LOADER_WAIT.push(p);
})();
