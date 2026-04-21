'use strict';

const path = require('node:path');
const fs = require('node:fs/promises');
const YAML = require('yaml');

const DEFAULT_SITE_ORIGIN = 'https://mascari4615.github.io';

function shouldExclude(relFromPostsDir, basename) {
  const posix = relFromPostsDir.replace(/\\/g, '/');
  if (posix.includes('/_drafts/') || posix.startsWith('_drafts/')) {
    return true;
  }
  if (posix.includes('/ticket/') || posix.startsWith('ticket/')) {
    return true;
  }
  const lower = basename.toLowerCase();
  if (lower.endsWith('-draft.md') || lower.endsWith('-template.md')) {
    return true;
  }
  if (/draft/i.test(basename)) {
    return true;
  }
  return false;
}

async function walkMdFiles(dir, base = dir) {
  const out = [];
  const ents = await fs.readdir(dir, { withFileTypes: true });
  for (const ent of ents) {
    const full = path.join(dir, ent.name);
    const rel = path.relative(base, full);
    if (ent.isDirectory()) {
      out.push(...(await walkMdFiles(full, base)));
    } else if (ent.isFile() && ent.name.endsWith('.md')) {
      if (!shouldExclude(rel, ent.name)) {
        out.push(full);
      }
    }
  }
  return out;
}

function slugFromBasename(name) {
  const m = name.match(/^(\d{4}-\d{2}-\d{2})-(.+)\.md$/);
  return m ? m[2] : path.basename(name, '.md');
}

function parseFrontMatter(raw) {
  if (!raw.startsWith('---')) {
    return { data: {}, body: raw };
  }
  const lines = raw.split('\n');
  if (lines[0].trim() !== '---') {
    return { data: {}, body: raw };
  }
  let end = -1;
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      end = i;
      break;
    }
  }
  if (end === -1) {
    return { data: {}, body: raw };
  }
  const fmText = lines.slice(1, end).join('\n');
  let data = {};
  try {
    data = YAML.parse(fmText) || {};
  } catch {
    data = {};
  }
  const body = lines.slice(end + 1).join('\n');
  return { data, body };
}

function slugFromHref(href, siteOrigin) {
  if (!href || href.startsWith('#')) {
    return null;
  }
  const t = href.trim();
  if (t.includes('/_posts/')) {
    return null;
  }
  try {
    const base = siteOrigin.replace(/\/$/, '');
    const u = new URL(t, `${base}/`);
    const segs = u.pathname.split('/').filter(Boolean);
    const idx = segs.indexOf('posts');
    if (idx < 0 || idx >= segs.length - 1) {
      return null;
    }
    let slug = segs[idx + 1];
    slug = slug.replace(/\.(md|html)$/i, '');
    slug = slug.replace(/\.+$/, '');
    return slug || null;
  } catch {
    return null;
  }
}

const MD_LINK_RE = /\[([^\]]*)\]\(([^)]+)\)/g;

function extractMarkdownHrefs(body) {
  const urls = [];
  let m;
  MD_LINK_RE.lastIndex = 0;
  while ((m = MD_LINK_RE.exec(body)) !== null) {
    urls.push(m[2].trim());
  }
  return urls;
}

async function buildJekyllPostGraph(root, siteOrigin = DEFAULT_SITE_ORIGIN) {
  const postsDir = path.join(root, '_posts');
  const files = await walkMdFiles(postsDir, postsDir);
  const bySlug = new Map();

  for (const file of files) {
    const raw = await fs.readFile(file, 'utf8');
    const { data } = parseFrontMatter(raw);
    const basename = path.basename(file);
    const slug = slugFromBasename(basename);
    const title = typeof data.title === 'string' ? data.title : slug;
    let categories = data.categories;
    if (!Array.isArray(categories)) {
      categories = categories != null ? [categories] : [];
    }
    const tags = Array.isArray(data.tags) ? data.tags : data.tags != null ? [data.tags] : [];
    const group = categories[0] != null ? String(categories[0]) : '';

    bySlug.set(slug, {
      id: slug,
      label: title,
      href: `/posts/${slug}/`,
      ...(group ? { group } : {}),
      meta: {
        categories,
        tags,
        file: path.relative(root, file).replace(/\\/g, '/')
      }
    });
  }

  const linkSet = new Set();
  const links = [];

  for (const file of files) {
    const raw = await fs.readFile(file, 'utf8');
    const { body } = parseFrontMatter(raw);
    const slug = slugFromBasename(path.basename(file));
    for (const href of extractMarkdownHrefs(body)) {
      const target = slugFromHref(href, siteOrigin);
      if (!target || target === slug || !bySlug.has(target)) {
        continue;
      }
      const key = `${slug}->${target}`;
      if (linkSet.has(key)) {
        continue;
      }
      linkSet.add(key);
      links.push({ source: slug, target, kind: 'link' });
    }
  }

  return { version: 1, nodes: [...bySlug.values()], links };
}

module.exports = { buildJekyllPostGraph, DEFAULT_SITE_ORIGIN };
