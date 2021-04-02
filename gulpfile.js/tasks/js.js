#!/usr/bin/env node

"use strict";

const { src, dest, watch, series, parallel} = require('gulp');

const concat = require('gulp-concat');
const rename = require("gulp-rename");
const uglify = require('gulp-uglify');
const insert = require('gulp-insert');
const fs = require('fs');

const JS_ROOT = 'assets/js';
const jsDest = `${ JS_ROOT }/dist/`;

function concatJs(files, output) {
  return src(files)
    .pipe(concat(output))
    .pipe(rename({ extname: '.min.js' }))
    .pipe(dest(jsDest));
}

function minifyJs() {
  return src(`${ jsDest }/*.js`)
    .pipe(insert.prepend(fs.readFileSync(`${ JS_ROOT }/_copyright`, 'utf8')))
    .pipe(uglify({output: {comments: /^!|@preserve|@license|@cc_on/i}}))
    .pipe(dest(jsDest));
}

const homeJs = () => {
  return concatJs([
      `${JS_ROOT}/_commons/*.js`,
      `${JS_ROOT}/_utils/timeago.js`
    ],
    'home'
  );
};

const postJs = () => {
  return concatJs([
      `${JS_ROOT}/_commons/*.js`,
      `${JS_ROOT}/_utils/timeago.js`,
      `${JS_ROOT}/_utils/img-hyperlink.js`,
      `${JS_ROOT}/_utils/lang-badge.js`,
      // 'smooth-scroll.js' must be called after ToC is ready
      `${JS_ROOT}/_utils/smooth-scroll.js`
    ], 'post'
  );
};

const categoriesJs = () => {
  return concatJs([
      `${JS_ROOT}/_commons/*.js`,
      `${JS_ROOT}/_utils/category-collapse.js`
    ], 'categories'
  );
};

const pageJs = () => {
  return concatJs([
      `${JS_ROOT}/_commons/*.js`,
      `${JS_ROOT}/_utils/smooth-scroll.js`
    ], 'page'
  );
};

// GA pageviews report
const pvreportJs = () => {
  return concatJs(`${JS_ROOT}/_utils/pageviews.js`, 'pvreport');
};

const buildJs = parallel(homeJs, postJs, categoriesJs, pageJs, pvreportJs);

exports.build = series(buildJs, minifyJs);

exports.liveRebuild = () => {
  buildJs();

  watch([
      `${ JS_ROOT }/_commons/*.js`,
      `${ JS_ROOT }/_utils/*.js`,
      `${ JS_ROOT }/lib/*.js`
    ],
    buildJs
  )
}

