#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic invokes all initial scripts for project.
v2.0
https://github.com/cotes2020/jekyll-theme-chirpy
© 2018-2019 Cotes Chung
Licensed under MIT
"""

import update_posts_lastmod
import pages_generator


update_posts_lastmod

pages_generator
