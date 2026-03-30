/**
 * Emit browser IIFE scripts from src/ into js/ (mirrors paths under src/).
 * bundle: true so type-only imports resolve and shared types from types/ are inlined/erased.
 */
import * as esbuild from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = __dirname;

const entryPoints = [
  'src/widgets/imageconvert/core.ts',
  'src/widgets/imageconvert/batch-pipeline.ts',
  'src/widgets/randomgen/randomgen-color.ts',
  'src/widgets/randomgen/randomgen-time.ts',
  'src/widgets/randomgen/randomgen-number.ts',
  'src/widgets/randomgen/randomgen-name.ts',
  'src/widgets/randomgen/randomgen-topics.ts',
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
