#!/usr/bin/env node
/**
 * 텍스트 파일(한 줄 = 한 후보 이름) → 티어리스트 후보 풀 catalog JSON (version 2).
 *
 *   node apps/karmolab/scripts/text-to-tierlist-catalog.mjs -i .\apps\karmolab\data\tierlists\anime-titles-ko-edit-karmoddrine.txt -o .\apps\karmolab\data\tierlists\karmo-anime.json -t karmo-anime --sync .\apps\karmolab\data\tierlists\karmo-anime.json 
 *   type titles.txt | node apps/karmolab/scripts/text-to-tierlist-catalog.mjs -o out.json -t my-anime
 *
 * -t / --title: catalog title (생략 시 --sync 는 기존 JSON 제목 유지, 그 외 imported-catalog)
 * --merge 기존.json: 기존 items 전부 유지 + 텍스트에만 있는 이름은 새 id로 추가
 * --sync 기존.json: 텍스트 기준 동기화 — 이름이 같으면 기존 항목(id·imageKey 등) 그대로, 텍스트에 없으면 삭제, 새 이름은 새 id
 * --no-dedupe: 같은 이름 여러 줄이면 항목 여러 개 (merge/sync 시 이름이 맞는 기존 항목을 순서대로 소비)
 */

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { stdout } from 'node:process';

const ALNUM = 'abcdefghijklmnopqrstuvwxyz0123456789';

function newItemId() {
    const buf = randomBytes(16);
    let s = '';
    for (let i = 0; i < 14; i++) s += ALNUM[buf[i] % 36];
    return `ti-${s}`;
}

function parseArgs() {
    const a = process.argv.slice(2);
    const out = {
        input: null,
        output: null,
        title: undefined,
        merge: null,
        sync: null,
        dedupe: true,
    };
    for (let i = 0; i < a.length; i++) {
        if ((a[i] === '-i' || a[i] === '--input') && a[i + 1]) { out.input = a[++i]; continue; }
        if ((a[i] === '-o' || a[i] === '--output') && a[i + 1]) { out.output = a[++i]; continue; }
        if ((a[i] === '-t' || a[i] === '--title') && a[i + 1]) { out.title = a[++i]; continue; }
        if (a[i] === '--merge' && a[i + 1]) { out.merge = a[++i]; continue; }
        if (a[i] === '--sync' && a[i + 1]) { out.sync = a[++i]; continue; }
        if (a[i] === '--no-dedupe') { out.dedupe = false; continue; }
        if (a[i] === '-h' || a[i] === '--help') {
            stdout.write(
                '사용법:\n' +
                    '  -i 입력.txt -o 출력.json [-t 제목] [--merge 기존.json | --sync 기존.json] [--no-dedupe]\n' +
                    '  --merge  기존 항목 유지 + 텍스트에만 있는 이름 추가\n' +
                    '  --sync   텍스트와 동기화(기존 id·필드 보존, 빠진 이름 삭제, 새 줄 추가)\n',
            );
            process.exit(0);
        }
    }
    return out;
}

function readStdinUtf8() {
    try {
        return readFileSync(0, 'utf8');
    } catch {
        return '';
    }
}

function linesFromText(raw) {
    return raw
        .split(/\r?\n/)
        .map(l => l.trim())
        .filter(l => l.length > 0);
}

function uniqueId(existingItems) {
    let id;
    do {
        id = newItemId();
    } while (existingItems[id]);
    return id;
}

function itemNameKey(item) {
    return String(item?.name ?? '').trim();
}

function dedupeNamesPreserveOrder(names) {
    const seen = new Set();
    const out = [];
    for (const n of names) {
        if (seen.has(n)) continue;
        seen.add(n);
        out.push(n);
    }
    return out;
}

/** --sync: 텍스트 줄 순서대로 items 재구성. 이름 일치 시 기존 항목 전체 복사( id 유지 ), 없으면 새 id */
function syncItemsFromText(oldItems, rawNames, dedupe) {
    const names = dedupe ? dedupeNamesPreserveOrder(rawNames) : rawNames;
    const queues = new Map();
    for (const id of Object.keys(oldItems)) {
        const item = oldItems[id];
        const key = itemNameKey(item);
        if (!queues.has(key)) queues.set(key, []);
        queues.get(key).push({ id, item });
    }

    const out = {};
    for (const name of names) {
        const q = queues.get(name);
        if (q && q.length > 0) {
            const { id, item } = q.shift();
            const copy = JSON.parse(JSON.stringify(item));
            copy.id = id;
            out[id] = copy;
        } else {
            const id = uniqueId(out);
            out[id] = { id, name };
        }
    }
    return out;
}

function mergeItems(existingItems, names, dedupe) {
    const items = { ...existingItems };
    const seen = new Set(dedupe ? Object.values(items).map(it => itemNameKey(it)) : []);

    for (const name of names) {
        if (dedupe && seen.has(name)) continue;
        const id = uniqueId(items);
        items[id] = { id, name };
        if (dedupe) seen.add(name);
    }
    return items;
}

function pruneImages(images, itemIds) {
    if (!images || typeof images !== 'object') return null;
    const pruned = {};
    for (const id of itemIds) {
        if (Object.prototype.hasOwnProperty.call(images, id)) pruned[id] = images[id];
    }
    return Object.keys(pruned).length ? pruned : null;
}

function loadCatalogJson(path, label) {
    if (!existsSync(path)) {
        console.error(`${label} 파일 없음: ${path}`);
        process.exit(1);
    }
    const prev = JSON.parse(readFileSync(path, 'utf8'));
    if (prev.version !== 2 || prev.kind !== 'catalog' || !prev.items || typeof prev.items !== 'object') {
        console.error(`${label} 대상은 version:2, kind:catalog, items 객체 필요`);
        process.exit(1);
    }
    return prev;
}

function main() {
    const { input, output, title, merge, sync, dedupe } = parseArgs();
    if (!output) {
        console.error('-o 출력.json 필요');
        process.exit(1);
    }
    if (merge && sync) {
        console.error('--merge 와 --sync 는 함께 쓸 수 없습니다.');
        process.exit(1);
    }

    let raw;
    if (input) {
        if (!existsSync(input)) {
            console.error(`파일 없음: ${input}`);
            process.exit(1);
        }
        raw = readFileSync(input, 'utf8');
    } else {
        raw = readStdinUtf8();
    }
    if (!raw.trim()) {
        console.error('입력 없음 (-i 파일 또는 stdin)');
        process.exit(1);
    }

    const names = linesFromText(raw);
    let prev = null;
    let items;

    if (sync) {
        prev = loadCatalogJson(sync, '--sync');
        items = syncItemsFromText(prev.items, names, dedupe);
    } else if (merge) {
        prev = loadCatalogJson(merge, '--merge');
        items = mergeItems(prev.items, names, dedupe);
    } else {
        items = mergeItems({}, names, dedupe);
    }

    const catalogTitle =
        title !== undefined ? title : prev?.title !== undefined && prev.title !== '' ? prev.title : 'imported-catalog';

    const catalog = {
        version: 2,
        kind: 'catalog',
        title: catalogTitle,
        category: typeof prev?.category === 'string' ? prev.category : '',
        items,
    };

    const pruned = pruneImages(prev?.images, Object.keys(items));
    if (pruned) catalog.images = pruned;

    writeFileSync(output, `${JSON.stringify(catalog, null, 2)}\n`, 'utf8');
    const mode = sync ? 'sync' : merge ? 'merge' : 'new';
    console.error(`저장: ${output} (${mode}, 항목 ${Object.keys(items).length}개)`);
}

main();
