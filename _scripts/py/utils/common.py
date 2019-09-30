#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Common functions to other scripts.

© 2018-2019 Cotes Chung
MIT License
'''

import sys


def get_yaml(path):
    """
    Return the Yaml block of a post and the linenumbers of it.
    """
    end = False
    yaml = ""
    num = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() == '---':
                if end:
                    break
                else:
                    end = True
                    continue
            else:
                num += 1

            yaml += line

    return yaml, num


def check_py_version():
    if not sys.version_info.major == 3 and sys.version_info.minor >= 5:
        print("WARNING: This script requires Python 3.5 or higher, "
              "however you are using Python {}.{}."
              .format(sys.version_info.major, sys.version_info.minor))
        sys.exit(1)
