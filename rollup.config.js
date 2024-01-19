import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import license from 'rollup-plugin-license';
import path from 'path';

const JS_SRC = '_javascript';
const JS_DIST = 'assets/js/dist';
const isProd = process.env.NODE_ENV === 'production';

function build(filename) {
  return {
    input: [`${JS_SRC}/${filename}.js`],
    output: {
      file: `${JS_DIST}/${filename}.min.js`,
      format: 'iife',
      name: 'Chirpy',
      sourcemap: !isProd
    },
    watch: {
      include: `${JS_SRC}/**`
    },
    plugins: [
      babel({
        babelHelpers: 'bundled',
        presets: ['@babel/env'],
        plugins: ['@babel/plugin-proposal-class-properties']
      }),
      license({
        banner: {
          commentStyle: 'ignored',
          content: { file: path.join(__dirname, JS_SRC, '_copyright') }
        }
      }),
      isProd && terser()
    ]
  };
}

export default [
  build('commons'),
  build('home'),
  build('categories'),
  build('page'),
  build('post'),
  build('misc')
];
