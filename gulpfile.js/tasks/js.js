#!/usr/bin/env node

"use strict";

const { src, dest, watch, series, parallel} = require('gulp');

const concat = require('gulp-concat');
const rename = require("gulp-rename");
const uglify = require('gulp-uglify');
const insert = require('gulp-insert');
const fs = require('fs');

const JS_SRC = '_javascript';
const JS_DEST = `assets/js/dist/`;

function concatJs(files, output) {
  return src(files)
    .pipe(concat(output))
    .pipe(rename({ extname: '.min.js' }))
    .pipe(dest(JS_DEST));
}

function minifyJs() {
  return src(`${ JS_DEST }/*.js`)
    .pipe(insert.prepend(fs.readFileSync(`${ JS_SRC }/copyright`, 'utf8')))
    .pipe(uglify({output: {comments: /^!|@preserve|@license|@cc_on/i}}))
    .pipe(dest(JS_DEST));
}

<<<<<<< HEAD
=======
const commonsJs = () => {
  return concatJs(`${JS_SRC}/commons/*.js`, 'commons');
};

>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
const homeJs = () => {
  return concatJs([
      `${JS_SRC}/commons/*.js`,
      `${JS_SRC}/utils/timeago.js`
    ],
    'home'
  );
};

const postJs = () => {
  return concatJs([
      `${JS_SRC}/commons/*.js`,
<<<<<<< HEAD
      `${JS_SRC}/utils/timeago.js`,
      `${JS_SRC}/utils/lang-badge.js`,
=======
      `${JS_SRC}/utils/img-extra.js`,
      `${JS_SRC}/utils/timeago.js`,
      `${JS_SRC}/utils/checkbox.js`,
      `${JS_SRC}/utils/clipboard.js`,
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
      // 'smooth-scroll.js' must be called after ToC is ready
      `${JS_SRC}/utils/smooth-scroll.js`
    ], 'post'
  );
};

const categoriesJs = () => {
  return concatJs([
      `${JS_SRC}/commons/*.js`,
      `${JS_SRC}/utils/category-collapse.js`
    ], 'categories'
  );
};

const pageJs = () => {
  return concatJs([
      `${JS_SRC}/commons/*.js`,
<<<<<<< HEAD
      `${JS_SRC}/utils/smooth-scroll.js`
=======
      `${JS_SRC}/utils/checkbox.js`,
      `${JS_SRC}/utils/img-extra.js`,
      `${JS_SRC}/utils/clipboard.js`
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
    ], 'page'
  );
};

// GA pageviews report
const pvreportJs = () => {
  return concatJs(`${JS_SRC}/utils/pageviews.js`, 'pvreport');
};

<<<<<<< HEAD
const buildJs = parallel(homeJs, postJs, categoriesJs, pageJs, pvreportJs);
=======
const buildJs = parallel(commonsJs, homeJs, postJs, categoriesJs, pageJs, pvreportJs);
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba

exports.build = series(buildJs, minifyJs);

exports.liveRebuild = () => {
  buildJs();

  watch([
      `${ JS_SRC }/commons/*.js`,
      `${ JS_SRC }/utils/*.js`,
      `${ JS_SRC }/lib/*.js`
    ],
    buildJs
<<<<<<< HEAD
  )
}

=======
  );
};
>>>>>>> ebb3dc940c22d864dc41a16f1d84c1a0c0a003ba
