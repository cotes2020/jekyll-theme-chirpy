import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import fs from 'fs';
import pkg from './package.json';

const SRC_DEFAULT = '_javascript';
const SRC_PWA = `${SRC_DEFAULT}/pwa`;
const SRC_GRAPH = `${SRC_DEFAULT}/graph-view`;
const DIST = 'assets/js/dist';
const GRAPH_DIST = 'assets/js/graph-view';

const banner = `/*!
 * ${pkg.name} v${pkg.version} | © ${pkg.since} ${pkg.author} | ${pkg.license} Licensed | ${pkg.homepage}
 */`;
const frontmatter = '---\npermalink: /:basename\n---\n';
const isProd = process.env.BUILD === 'production';

let hasWatched = false;

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

function build(
  filename,
  { src = SRC_DEFAULT, jekyll = false, outputName = null } = {}
) {
  const tsFile = `${src}/${filename}.ts`;
  const jsFile = `${src}/${filename}.js`;
  const input = fs.existsSync(tsFile) ? tsFile : jsFile;

  const shouldWatch = hasWatched ? false : true;

  if (!hasWatched) {
    hasWatched = true;
  }

  return {
    input,
    output: {
      file: `${DIST}/${filename}.min.js`,
      format: 'iife',
      ...(outputName !== null && { name: outputName }),
      banner,
      sourcemap: !isProd && !jekyll
    },
    ...(shouldWatch && { watch: { include: `${SRC_DEFAULT}/**/*.{js,ts}` } }),
    plugins: [
      typescript(),
      babel({
        babelHelpers: 'bundled',
        presets: ['@babel/env'],
        plugins: [
          '@babel/plugin-transform-class-properties',
          '@babel/plugin-transform-private-methods'
        ],
        extensions: ['.js', '.ts']
      }),
      nodeResolve(),
      isProd && terser(),
      jekyll && insertFrontmatter()
    ]
  };
}

/**
 * Post graph tab: ES module bundle, d3 loaded from CDN (external URL).
 */
function buildPostGraph() {
  return {
    input: `${SRC_GRAPH}/bootstrap-post-graph.ts`,
    output: {
      file: `${GRAPH_DIST}/bootstrap-post-graph.js`,
      format: 'es',
      sourcemap: !isProd
    },
    watch: {
      include: `${SRC_GRAPH}/**/*.{ts,js}`
    },
    plugins: [
      typescript(),
      babel({
        babelHelpers: 'bundled',
        presets: ['@babel/env'],
        plugins: [
          '@babel/plugin-transform-class-properties',
          '@babel/plugin-transform-private-methods'
        ],
        extensions: ['.js', '.ts']
      }),
      nodeResolve(),
      isProd && terser()
    ],
    external: (id) => /^https?:\/\//.test(id)
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
  build('theme', { outputName: 'Theme' }),
  build('app', { src: SRC_PWA, jekyll: true }),
  build('sw', { src: SRC_PWA, jekyll: true }),
  buildPostGraph()
];
