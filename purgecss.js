const fs = require('fs').promises;
const { PurgeCSS } = require('purgecss');
const DIST_PATH = '_sass/vendors';
const output = `${DIST_PATH}/_bootstrap.scss`;

const config = {
  content: ['_includes/**/*.html', '_layouts/**/*.html', '_javascript/**/*.js'],
  css: ['node_modules/bootstrap/dist/css/bootstrap.min.css'],
  keyframes: true,
  variables: true,
  // The `safelist` should be changed appropriately for future development
  safelist: {
    standard: [/^collaps/, /^w-/, 'shadow', 'border', 'kbd'],
    greedy: [/^col-/, /tooltip/]
  }
};

function main() {
  fs.rm(DIST_PATH, { recursive: true, force: true })
    .then(() => fs.mkdir(DIST_PATH))
    .then(() => new PurgeCSS().purge(config))
    .then((result) => {
      return fs.writeFile(output, result[0].css);
    })
    .catch((err) => {
      console.error('Error during PurgeCSS process:', err);
    });
}

main();
