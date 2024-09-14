import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import fs from 'fs';
import pkg from './package.json';

const SRC_DEFAULT = '_javascript';
const SRC_PWA = `${SRC_DEFAULT}/pwa`;
const DIST = 'assets/js/dist';

const banner = `/*!
 * ${pkg.name} v${pkg.version} | © ${pkg.since} ${pkg.author} | ${pkg.license} Licensed | ${pkg.homepage}
 */`;

const frontmatter = `---\npermalink: /:basename\n---\n`;

const isProd = process.env.BUILD === 'production';

function cleanup() {
  fs.rmSync(DIST, { recursive: true, force: true });
  console.log(`> Directory "${DIST}" has been cleaned.`);
}

function insertFrontmatter() {
  return {
    name: 'insert-frontmatter',
    generateBundle(_, bundle) {
      for (const chunkOrAsset of Object.values(bundle)) {
        if (chunkOrAsset.type === 'chunk') {
          chunkOrAsset.code = frontmatter + chunkOrAsset.code;
        }
      }
    }
  };
}

function build(filename, { src = SRC_DEFAULT, jekyll = false } = {}) {
  return {
    input: `${src}/${filename}.js`,
    output: {
      file: `${DIST}/${filename}.min.js`,
      format: 'iife',
      name: 'Chirpy',
      banner,
      sourcemap: !isProd && !jekyll
    },
    watch: {
      include: `${src}/**`
    },
    plugins: [
      babel({
        babelHelpers: 'bundled',
        presets: ['@babel/env'],
        plugins: ['@babel/plugin-transform-class-properties']
      }),
      nodeResolve(),
      isProd && terser(),
      jekyll && insertFrontmatter()
    ]
  };
}

cleanup();

export default [
  build('commons'),
  build('home'),
  build('categories'),
  build('page'),
  build('post'),
  build('misc'),
  build('app', { src: SRC_PWA, jekyll: true }),
  build('sw', { src: SRC_PWA, jekyll: true })
];
