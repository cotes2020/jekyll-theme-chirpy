/**
 * Emit browser scripts from src/ into js/ (mirrors paths under src/), and src/world → ../world/ (wiki loaders).
 * - Most entries: bundle + iife (type-only imports resolve).
 * - mdd.ts / gemini.ts / toolbox.ts: bundle false + esm so top-level globals stay visible (no extra IIFE).
 */
import * as esbuild from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = __dirname;

await esbuild.build({
  entryPoints: [join(root, 'src/mdd.ts')],
  outfile: join(root, 'js/mdd.js'),
  bundle: false,
  format: 'esm',
  platform: 'browser',
  target: ['es2020'],
  logLevel: 'info'
});

await esbuild.build({
  entryPoints: [join(root, 'src/gemini.ts')],
  outfile: join(root, 'js/gemini.js'),
  bundle: false,
  format: 'esm',
  platform: 'browser',
  target: ['es2020'],
  logLevel: 'info'
});

await esbuild.build({
  entryPoints: [join(root, 'src/toolbox.ts')],
  outfile: join(root, 'js/toolbox.js'),
  bundle: false,
  format: 'esm',
  platform: 'browser',
  target: ['es2020'],
  logLevel: 'info'
});

const entryPoints = [
  'src/widgets/imageconvert/core.ts',
  'src/widgets/imageconvert/batch-pipeline.ts',
  'src/widgets/imageconvert/imageconvert.ts',
  'src/widgets/imageconvert/widget.ts',
  'src/widgets/randomgen/randomgen-color.ts',
  'src/widgets/randomgen/randomgen-time.ts',
  'src/widgets/randomgen/randomgen-number.ts',
  'src/widgets/randomgen/randomgen-name.ts',
  'src/widgets/randomgen/randomgen-topics.ts',
  'src/widgets/randomgen/randomgen.ts',
  'src/widgets/toast.ts',
  'src/widgets/imageedit.ts',
  'src/widgets/crypto.ts',
  'src/widgets/memo.ts',
  'src/widgets/imagelib.ts',
  'src/widgets/conch.ts',
  'src/widgets/postgraph.ts',
  'src/widgets/fortune.ts',
  'src/widgets/gacha.ts',
  'src/widgets/bounce.ts',
  'src/widgets/bubble.ts',
  'src/widgets/countdown.ts',
  'src/widgets/darkroom.ts',
  'src/widgets/dashboard.ts',
  'src/widgets/devtools.ts',
  'src/widgets/eyes.ts',
  'src/widgets/favorites.ts',
  'src/widgets/folder.ts',
  'src/widgets/font.ts',
  'src/widgets/hacker.ts',
  'src/widgets/hourglass.ts',
  'src/widgets/moon.ts',
  'src/widgets/morse.ts',
  'src/widgets/news.ts',
  'src/widgets/particle.ts',
  'src/widgets/password.ts',
  'src/widgets/pet.ts',
  'src/widgets/reaction.ts',
  'src/widgets/servermonitor.ts',
  'src/widgets/shylink.ts',
  'src/widgets/speed.ts',
  'src/widgets/stone.ts',
  'src/widgets/user.ts',
  'src/widgets/youtubedl.ts',
  'src/widgets/chatbot/characters.ts',
  'src/widgets/chatbot/chatbot.ts',
  'src/widgets/chatbot/karmo-image.ts',
  'src/widgets/chatbot/markdown.ts',
  'src/widgets/chatbot/prompt.ts',
  'src/widgets/chatbot/styles.ts',
  'src/widgets/docs/docs.ts',
  'src/widgets/imagegen/config.ts',
  'src/widgets/imagegen/core.ts',
  'src/widgets/imagegen/imagegen.ts',
  'src/widgets/imagegen/presets.ts',
  'src/widgets/imagegen/styles.ts',
  'src/widgets/linktree/linktree.ts',
  'src/widgets/planner/planner.ts',
  'src/widgets/tierlist/dialogs.ts',
  'src/widgets/tierlist/dnd.ts',
  'src/widgets/tierlist/index.ts',
  'src/widgets/tierlist/namespace.ts',
  'src/widgets/tierlist/publish.ts',
  'src/widgets/tierlist/render.ts',
  'src/widgets/tierlist/storage.ts',
  'src/widgets/tierlist/styles.ts',
  'src/widgets/tierlist/tierlist.ts',
  'src/widgets/tierlist/ui.ts',
  'src/widgets/worldwiki/worldwiki.ts',
  'src/widgets-manifest.ts',
  'src/widgets-lazy-meta.ts',
  'src/widgets-loader.ts'
];

for (const rel of entryPoints) {
  const outfile = rel.replace(/^src\//, 'js/').replace(/\.ts$/, '.js');
  await esbuild.build({
    entryPoints: [join(root, rel)],
    outfile: join(root, outfile),
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: ['es2020'],
    logLevel: 'info'
  });
}

const worldEntryPoints = [
  'src/world/world.ts',
  'src/world/parse-md.ts',
  'src/world/load-characters-from-wiki.ts'
];
for (const rel of worldEntryPoints) {
  const outfile = rel.replace(/^src\/world\//, 'world/').replace(/\.ts$/, '.js');
  await esbuild.build({
    entryPoints: [join(root, rel)],
    outfile: join(root, outfile),
    bundle: true,
    format: 'iife',
    platform: 'browser',
    target: ['es2020'],
    logLevel: 'info'
  });
}
