import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import license from 'rollup-plugin-license';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import fs from 'fs';
import path from 'path';
import yaml from '@rollup/plugin-yaml';

const SRC_DEFAULT = '_javascript';
const DIST_DEFAULT = 'assets/js/dist';
const SRC_PWA = `${SRC_DEFAULT}/pwa`;

const isProd = process.env.BUILD === 'production';

if (fs.existsSync(DIST_DEFAULT)) {
  fs.rm(DIST_DEFAULT, { recursive: true, force: true }, (err) => {
    if (err) {
      throw err;
    }
  });
}

function build(filename, opts = {}) {
  const src = opts.src || SRC_DEFAULT;
  const dist = opts.dist || DIST_DEFAULT;
  const bannerUrl =
    opts.bannerUrl || path.join(__dirname, SRC_DEFAULT, '_copyright');
  const commentStyle = opts.commentStyle || 'ignored';

  return {
    input: [`${src}/${filename}.js`],
    output: {
      file: `${dist}/${filename}.min.js`,
      format: 'iife',
      name: 'Chirpy',
      sourcemap: !isProd
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
      yaml(),
      isProd && commentStyle === 'none' && terser(),
      license({
        banner: {
          commentStyle,
          content: { file: bannerUrl }
        }
      }),
      isProd && commentStyle !== 'none' && terser()
    ]
  };
}

export default [
  build('commons'),
  build('home'),
  build('categories'),
  build('page'),
  build('post'),
  build('misc'),
  build('app', { src: SRC_PWA }),
  build('sw', {
    src: SRC_PWA,
    bannerUrl: path.join(__dirname, SRC_PWA, '_frontmatter'),
    commentStyle: 'none'
  })
];
