#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update (or create if not existed) field 'seo.date_modified'
in posts' Front Matter by their latest git commit date.

Dependencies:
  - git
  - ruamel.yaml

© 2018-2019 Cotes Chung
Licensed under MIT
"""

import sys
import glob
import os
import subprocess
import shutil

from ruamel.yaml import YAML
from utils.common import get_yaml
from utils.common import check_py_version


POSTS_PATH = "_posts"


def help():
    print("Usage: "
          "   python update_posts_lastmod.py [option]\n"
          "Options:\n"
          "   -v, --verbose      Print verbose logs\n")


def update_lastmod(verbose):
    count = 0
    yaml = YAML()

    for post in glob.glob(os.path.join(POSTS_PATH, "*.md")):
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

        frontmatter, line_num = get_yaml(post)
        meta = yaml.load(frontmatter)

        if 'seo' in meta:
            if ('date_modified' in meta['seo'] and
                    meta['seo']['date_modified'] == git_lastmod):
                continue
            else:
                meta['seo']['date_modified'] = git_lastmod
        else:
            meta.insert(line_num, 'seo', dict(date_modified=git_lastmod))

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


def main():
    check_py_version()

    verbose = False

    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == sys.argv[0]:
                continue
            else:
                if arg == '-v' or arg == '--verbose':
                    verbose = True
                else:
                    help()
                    return

    update_lastmod(verbose)


main()
