/**
 * Emit browser scripts from src/ into js/ (mirrors paths under src/).
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
