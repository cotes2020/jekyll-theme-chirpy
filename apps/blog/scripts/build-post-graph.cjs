'use strict';

const path = require('node:path');
const fs = require('node:fs/promises');
const { normalizeGraph } = require('./graph/schema.cjs');
const { buildJekyllPostGraph } = require('./graph/adapters/jekyll-posts.cjs');

const root = path.join(__dirname, '..');
const outFile = path.join(root, 'assets/js/data/post-graph.json');

(async () => {
  const graph = normalizeGraph(await buildJekyllPostGraph(root));
  await fs.writeFile(outFile, `${JSON.stringify(graph)}\n`, 'utf8');
  console.log(
    `Wrote ${path.relative(root, outFile)} (${graph.nodes.length} nodes, ${graph.links.length} links)`
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
