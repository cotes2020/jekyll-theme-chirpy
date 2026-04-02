/**
 * KarmoWorld — Markdown frontmatter YAML (제한된 부분집합) 파싱
 * - 단일 파일: front matter --- ... --- + 본문 (`parseCharacterWikiMarkdown`)
 * - 분리 파일: `*.yaml`(메타) + `*.md`(본문만, `---` 없음) (`parseCharacterWikiFromSplitFiles`)
 * - 스칼라: key: value
 * - 블록 스칼라: key: | 다음 들여쓴 줄
 * - 목록: key: 다음 줄에 - item
 */
(function (): void {
  const K = window.KarmoWorld || {};
  window.KarmoWorld = K;

  function splitFrontmatter(md: string): { frontmatter: string; body: string } {
    if (!md.startsWith('---\n') && !md.startsWith('---\r\n')) return { frontmatter: '', body: md };
    const lines = md.split(/\r?\n/);
    if (lines[0].trim() !== '---') return { frontmatter: '', body: md };
    for (let i = 1; i < lines.length; i++) {
      const t = lines[i].trim();
      if (t === '---' || t === '...') {
        return {
          frontmatter: lines.slice(1, i).join('\n'),
          body: lines.slice(i + 1).join('\n').replace(/^\n+/, '')
        };
      }
    }
    return { frontmatter: '', body: md };
  }

  function unquoteScalar(s: string): string {
    let t = String(s).trim();
    if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
      return t.slice(1, -1).replace(/\\n/g, '\n');
    }
    return t;
  }

  /**
   * YAML 부분집합 (캐릭터 frontmatter용)
   */
  function parseYamlSimple(yaml: string): Record<string, unknown> {
    const lines = yaml.split(/\r?\n/);
    const obj: Record<string, unknown> = {};
    let i = 0;
    while (i < lines.length) {
      const raw = lines[i];
      if (/^\s*$/.test(raw) || /^\s*#/.test(raw)) {
        i++;
        continue;
      }
      const m = /^([a-zA-Z0-9_]+):\s*(.*)$/.exec(raw);
      if (!m) {
        i++;
        continue;
      }
      const key = m[1];
      let rest = m[2];
      if (rest === '|' || rest === '|+' || rest === '|-') {
        i++;
        const blockLines: string[] = [];
        let indent: number | null = null;
        while (i < lines.length) {
          const l = lines[i];
          const topKey = /^([a-zA-Z0-9_]+):\s*/.exec(l);
          if (topKey && !/^\s/.test(l)) break;
          if (l.trim() === '') {
            blockLines.push('');
            i++;
            continue;
          }
          if (/^\s/.test(l)) {
            const ind = l.match(/^(\s*)/);
            if (ind === null) {
              i++;
              continue;
            }
            if (indent === null) indent = ind[1].length;
            blockLines.push(l.slice(indent));
            i++;
            continue;
          }
          break;
        }
        obj[key] = blockLines.join('\n').replace(/\n+$/, '');
        continue;
      }
      if (rest === '') {
        i++;
        if (i < lines.length && /^\s*-\s/.test(lines[i])) {
          const list: string[] = [];
          while (i < lines.length && /^\s*-\s/.test(lines[i])) {
            list.push(lines[i].replace(/^\s*-\s*/, '').trim());
            i++;
          }
          obj[key] = list;
          continue;
        }
        obj[key] = '';
        continue;
      }
      obj[key] = unquoteScalar(rest);
      i++;
    }
    return obj;
  }

  function parseCharacterWikiMarkdown(md: string): { meta: Record<string, unknown>; body: string } {
    const { frontmatter, body } = splitFrontmatter(md);
    const meta = parseYamlSimple(frontmatter);
    return { meta, body };
  }

  function parseCharacterWikiFromSplitFiles(
    yamlText: string,
    mdText: string
  ): { meta: Record<string, unknown>; body: string } {
    return { meta: parseYamlSimple(yamlText), body: mdText.trim() };
  }

  K.parseMd = {
    splitFrontmatter,
    parseYamlSimple,
    parseCharacterWikiMarkdown,
    parseCharacterWikiFromSplitFiles
  };
})();
