// @ts-nocheck
/**
 * KarmoWorld — Markdown frontmatter YAML (제한된 부분집합) 파싱
 * - 본문과 분리: YAML --- ... ---
 * - 스칼라: key: value
 * - 블록 스칼라: key: | 다음 들여쓴 줄
 * - 목록: key: 다음 줄에 - item
 */
(function () {
    const K = window.KarmoWorld || {};
    window.KarmoWorld = K;

    function splitFrontmatter(md) {
        if (typeof md !== 'string') return { frontmatter: '', body: '' };
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

    function unquoteScalar(s) {
        s = String(s).trim();
        if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
            return s.slice(1, -1).replace(/\\n/g, '\n');
        }
        return s;
    }

    /**
     * YAML 부분집합 (캐릭터 frontmatter용)
     */
    function parseYamlSimple(yaml) {
        const lines = yaml.split(/\r?\n/);
        const obj = {};
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
                const blockLines = [];
                let indent = null;
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
                        if (indent === null) indent = l.match(/^(\s*)/)[1].length;
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
                    const list = [];
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

    function parseCharacterWikiMarkdown(md) {
        const { frontmatter, body } = splitFrontmatter(md);
        const meta = parseYamlSimple(frontmatter);
        return { meta, body };
    }

    K.parseMd = {
        splitFrontmatter,
        parseYamlSimple,
        parseCharacterWikiMarkdown
    };
})();
