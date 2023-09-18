#!/usr/bin/env node

"use strict";

const js = require('./tasks/js');

exports.default = js.build;

/* keep-alive develop mode, without uglify */
exports.dev = js.liveRebuild;
