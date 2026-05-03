/**
 * sync-wiki: memo → apps/karmolab/world/wiki/entities/ 단방향 sync.
 *
 * 정본: memo/ (사용자 직접 편집).
 * 출력: apps/karmolab/world/wiki/entities/<type-plural>/<slug>.{yaml,md}
 *
 * entity 타입 (정본 위치 → 출력 폴더):
 *   character → memo/characters/<dir>/{karmolab.yaml,wiki.md}  → entities/characters/
 *   system    → memo/systems/<slug>.md                          → entities/systems/
 *   concept   → memo/wm/design/concepts/<slug>.md (있으면)       → entities/concepts/
 *   lore      → memo/wm/design/lore/<slug>.md (있으면)           → entities/lore/
 *
 * 룰: 룰 정본 = memo/projects/karmolab/docs/entity-schema.md.
 */
import * as fs from 'node:fs';
import * as fsp from 'node:fs/promises';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KARMOLAB_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(KARMOLAB_ROOT, '../..');

// memo path: 환경변수 / 기본 사용자 경로 / repo 외부 — sync 는 dev/CI 환경 모두 대응.
const MEMO_PATH = process.env.KARMODDRINE_MEMO_PATH
  || path.resolve(REPO_ROOT, '../memo');

const OUT_ROOT = path.join(KARMOLAB_ROOT, 'world/wiki/entities');

const TYPE_OUT_DIR = {
  character: 'characters',
  system: 'systems',
  concept: 'concepts',
  lore: 'lore',
};

// ── frontmatter 파싱 (apps/karmolab/src/world/parse-md.ts 의 parseYamlSimple 와 같은 부분집합) ─────
function splitFrontmatter(md) {
  if (!md.startsWith('---\n') && !md.startsWith('---\r\n')) return { fm: '', body: md };
  const lines = md.split(/\r?\n/);
  if (lines[0].trim() !== '---') return { fm: '', body: md };
  for (let i = 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === '---' || t === '...') {
      return {
        fm: lines.slice(1, i).join('\n'),
        body: lines.slice(i + 1).join('\n').replace(/^\n+/, ''),
      };
    }
  }
  return { fm: '', body: md };
}

function unquoteScalar(s) {
  let t = String(s).trim();
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
    return t.slice(1, -1).replace(/\\n/g, '\n');
  }
  return t;
}

function parseYamlSimple(yaml) {
  const lines = yaml.split(/\r?\n/);
  const obj = {};
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    if (/^\s*$/.test(raw) || /^\s*#/.test(raw)) { i++; continue; }
    const m = /^([a-zA-Z0-9_]+):\s*(.*)$/.exec(raw);
    if (!m) { i++; continue; }
    const key = m[1];
    const rest = m[2];
    if (rest === '|' || rest === '|+' || rest === '|-') {
      i++;
      const blockLines = [];
      let indent = null;
      while (i < lines.length) {
        const l = lines[i];
        const topKey = /^([a-zA-Z0-9_]+):\s*/.exec(l);
        if (topKey && !/^\s/.test(l)) break;
        if (l.trim() === '') { blockLines.push(''); i++; continue; }
        if (/^\s/.test(l)) {
          const ind = l.match(/^(\s*)/);
          if (ind === null) { i++; continue; }
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

// ── yaml 직렬화 (parseYamlSimple 의 역. 부분집합) ────────────────────────────────────────────
function isMultilineBlock(value) {
  // newline 있을 때만 block scalar (|). single-line 은 길어도 single-line.
  // sync round-trip 일관성 우선 — parser 가 |/single 둘 다 같은 string 으로 파싱.
  return typeof value === 'string' && value.includes('\n');
}

function dumpYaml(obj, indent = 0) {
  const pad = '  '.repeat(indent);
  const out = [];
  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      if (value.length === 0) { out.push(`${pad}${key}: []`); continue; }
      const allScalar = value.every(v => typeof v !== 'object' || v === null);
      if (allScalar) {
        out.push(`${pad}${key}:`);
        for (const v of value) out.push(`${pad}  - ${String(v)}`);
      } else {
        // 객체 list (예: relationships, flow.nodes 등)
        out.push(`${pad}${key}:`);
        for (const v of value) {
          const entries = Object.entries(v);
          if (entries.length === 0) continue;
          const [firstKey, firstValue] = entries[0];
          out.push(`${pad}  - ${firstKey}: ${String(firstValue)}`);
          for (let i = 1; i < entries.length; i++) {
            out.push(`${pad}    ${entries[i][0]}: ${String(entries[i][1])}`);
          }
        }
      }
    } else if (typeof value === 'object') {
      out.push(`${pad}${key}:`);
      out.push(dumpYaml(value, indent + 1));
    } else if (isMultilineBlock(value)) {
      out.push(`${pad}${key}: |`);
      for (const line of String(value).split('\n')) out.push(`${pad}  ${line}`);
    } else {
      out.push(`${pad}${key}: ${String(value)}`);
    }
  }
  return out.join('\n');
}

// ── 검증 ────────────────────────────────────────────────────────────────────────────────────
function validateBase(meta, sourcePath) {
  const required = ['slug', 'entityId', 'type', 'title'];
  const missing = required.filter(k => !meta[k]);
  if (missing.length > 0) {
    throw new Error(`[${sourcePath}] 필수 필드 누락: ${missing.join(', ')}`);
  }
  if (!TYPE_OUT_DIR[meta.type]) {
    throw new Error(`[${sourcePath}] 알 수 없는 type: ${meta.type}. 허용: ${Object.keys(TYPE_OUT_DIR).join(' / ')}`);
  }
}

// ── 출력 ────────────────────────────────────────────────────────────────────────────────────
async function writeEntity(meta, body, sourcePath) {
  validateBase(meta, sourcePath);
  const outDir = path.join(OUT_ROOT, TYPE_OUT_DIR[meta.type]);
  await fsp.mkdir(outDir, { recursive: true });
  const slug = meta.slug;
  const yamlOut = dumpYaml(meta) + '\n';
  const mdOut = body.endsWith('\n') ? body : body + '\n';
  await fsp.writeFile(path.join(outDir, `${slug}.yaml`), yamlOut, 'utf8');
  await fsp.writeFile(path.join(outDir, `${slug}.md`), mdOut, 'utf8');
  return { slug, type: meta.type, outDir };
}

// ── walker: character (memo/characters/<dir>/) ─────────────────────────────────────────────
async function walkCharacters() {
  const charsRoot = path.join(MEMO_PATH, 'characters');
  if (!fs.existsSync(charsRoot)) return [];
  const out = [];
  const entries = await fsp.readdir(charsRoot, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(charsRoot, entry.name);
    const yamlFile = path.join(dir, 'karmolab.yaml');
    const mdFile = path.join(dir, 'wiki.md');
    if (!fs.existsSync(yamlFile)) continue;  // KarmoLab entity 가 아닌 캐릭터는 skip (yawnbot 만 사용)
    if (!fs.existsSync(mdFile)) {
      console.warn(`[character] ${entry.name}: karmolab.yaml 있지만 wiki.md 없음 — skip`);
      continue;
    }
    const yamlText = await fsp.readFile(yamlFile, 'utf8');
    const mdText = await fsp.readFile(mdFile, 'utf8');
    const meta = parseYamlSimple(yamlText);
    out.push(await writeEntity(meta, mdText, `memo/characters/${entry.name}/`));
  }
  return out;
}

// ── walker: system / concept / lore (단일 .md, frontmatter + body) ──────────────────────────
async function walkSingleType(type, memoSubpath) {
  const root = path.join(MEMO_PATH, memoSubpath);
  if (!fs.existsSync(root)) return [];
  const out = [];
  const entries = await fsp.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.md')) continue;
    if (entry.name.startsWith('.') || entry.name.startsWith('_')) continue;  // 숨김 / 임시
    if (entry.name.toLowerCase() === 'readme.md') continue;
    const filePath = path.join(root, entry.name);
    const text = await fsp.readFile(filePath, 'utf8');
    const { fm, body } = splitFrontmatter(text);
    if (!fm) {
      console.warn(`[${type}] ${entry.name}: frontmatter 없음 — skip`);
      continue;
    }
    const meta = parseYamlSimple(fm);
    if (meta.type && meta.type !== type) {
      console.warn(`[${type}] ${entry.name}: type 불일치 (${meta.type}) — skip`);
      continue;
    }
    if (!meta.type) meta.type = type;  // 디렉토리 위치로 type 추론
    out.push(await writeEntity(meta, body, `${memoSubpath}/${entry.name}`));
  }
  return out;
}

// ── main ────────────────────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`[sync-wiki] memo: ${MEMO_PATH}`);
  console.log(`[sync-wiki] out:  ${OUT_ROOT}`);
  if (!fs.existsSync(MEMO_PATH)) {
    console.error(`[sync-wiki] memo path 없음: ${MEMO_PATH}`);
    console.error('[sync-wiki] 환경변수 KARMODDRINE_MEMO_PATH 로 명시 가능.');
    process.exit(1);
  }

  const results = [];
  results.push(...await walkCharacters());
  results.push(...await walkSingleType('system', 'systems'));
  results.push(...await walkSingleType('concept', 'wm/design/concepts'));
  results.push(...await walkSingleType('lore', 'wm/design/lore'));

  console.log(`[sync-wiki] 처리: ${results.length} entity`);
  for (const r of results) {
    console.log(`  - ${r.type}/${r.slug}`);
  }
}

main().catch(err => {
  console.error('[sync-wiki] 실패:', err);
  process.exit(1);
});
