#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update (or create if not existed) field 'seo.date_modified'
in posts' Front Matter by their latest git commit date.

Dependencies:
  - git
  - ruamel.yaml

v2.0
https://github.com/cotes2020/jekyll-theme-chirpy
© 2018-2019 Cotes Chung
Licensed under MIT
"""

import sys
import glob
import os
import getopt
import subprocess
import shutil
import datetime
import time

from enum import Enum
from ruamel.yaml import YAML

from utils.common import get_yaml
from utils.common import check_py_version


Date = Enum('Date', ('GIT', 'FS'))

POSTS_PATH = '_posts'


def help():
    print("Usage: "
          "   python update_posts_lastmod.py [options]\n"
          "Options:\n"
          "   -f, --file  <file path>        Read a file.\n"
          "   -d, --dir   <directory path>   Read from a directory.\n"
          "   -h, --help                     Print help information\n"
          "   -v, --verbose                  Print verbose logs\n"
          "   -t, --datetime  < git | fs >   Chose post's datetime source, "
          "'git' for git-log, 'fs' for filesystem, default to 'git'.\n")


def update_lastmod(path, verbose, date):
    count = 0
    yaml = YAML()

    for post in glob.glob(path):

        lastmod = ''

        if date == Date.GIT:
            git_log_count = subprocess.getoutput(
                "git log --pretty=%ad \"{}\" | wc -l".format(post))

            if git_log_count == "1":
                continue

            git_lastmod = subprocess.getoutput(
                "git log -1 --pretty=%ad --date=iso \"{}\"".format(post))

            if not git_lastmod:
                continue

            lates_commit = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B', post]).decode('utf-8')

            if "[Automation]" in lates_commit and "Lastmod" in lates_commit:
                continue

            lastmod = git_lastmod

        elif date == Date.FS:
            t = os.path.getmtime(post)
            dt = datetime.datetime.fromtimestamp(t)
            lastmod = dt.strftime('%F %T') + time.strftime(' %z')

        frontmatter, line_num = get_yaml(post)
        meta = yaml.load(frontmatter)

        if 'seo' in meta:
            if ('date_modified' in meta['seo'] and
                    meta['seo']['date_modified'] == lastmod):
                continue
            else:
                meta['seo']['date_modified'] = lastmod
        else:
            meta.insert(line_num, 'seo', dict(date_modified=lastmod))

        output = 'new.md'
        if os.path.isfile(output):
            os.remove(output)

        with open(output, 'w', encoding='utf-8') as new, \
                open(post, 'r', encoding='utf-8') as old:
            new.write("---\n")
            yaml.dump(meta, new)
            new.write("---\n")
            line_num += 2

            lines = old.readlines()

            for line in lines:
                if line_num > 0:
                    line_num -= 1
                    continue
                else:
                    new.write(line)

        shutil.move(output, post)
        count += 1

        if verbose:
            print("[INFO] update 'lastmod' for:" + post)

    if count > 0:
        print("[INFO] Success to update lastmod for {} post(s).".format(count))


def main(argv):
    check_py_version()

    path = os.path.join(POSTS_PATH, "*.md")
    verbose = False
    date = Date.GIT

    try:
        opts, args = getopt.getopt(
            argv, "hf:d:vt:",
            ["file=", "dir=", "help", "verbose", "datetime="])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()

        elif opt == '-f' or opt == '--file':
            path = arg

        elif opt == '-d' or opt == '--dir':
            path = os.path.join(arg, "*.md")

        elif opt == '-v' or opt == '--verbose':
            verbose = True

        elif opt == '-t' or opt == '--datetime':
            if arg == 'git':
                date = Date.GIT
            elif arg == 'fs':
                date = Date.FS
            else:
                help()
                sys.exit(2)

    update_lastmod(path, verbose, date)


main(sys.argv[1:])
