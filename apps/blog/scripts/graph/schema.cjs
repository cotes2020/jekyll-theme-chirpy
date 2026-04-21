'use strict';

const GRAPH_SCHEMA_VERSION = 1;

function normalizeGraph(g) {
  const nodes = (g.nodes || []).map((n) => {
    const out = { id: n.id, label: n.label ?? n.id };
    if (n.href != null && n.href !== '') out.href = n.href;
    if (n.group != null && n.group !== '') out.group = n.group;
    if (n.meta != null && typeof n.meta === 'object' && Object.keys(n.meta).length) {
      out.meta = n.meta;
    }
    return out;
  });
  const seen = new Set();
  const links = [];
  for (const l of g.links || []) {
    if (!l || !l.source || !l.target) continue;
    const key = `${l.source}\t${l.target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const link = { source: l.source, target: l.target };
    if (l.kind) link.kind = l.kind;
    links.push(link);
  }
  return { version: g.version ?? GRAPH_SCHEMA_VERSION, nodes, links };
}

module.exports = { GRAPH_SCHEMA_VERSION, normalizeGraph };
