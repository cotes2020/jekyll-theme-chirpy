# -*- coding: utf-8 -*-

"""
Little script that replaces comment characters to html,
found in the *.hbs files.
"""

import os
import glob

if __name__ == '__main__':

    os.chdir(os.path.join(os.environ['HOME'],
                          "Downloads", "Casper-master"))
    for filename in glob.glob("*.hbs"):
        print(filename)

        # Read in the file
        with open(filename, 'r') as infile :
          filedata = infile.read()

        # Replace the target string
        # Replacing comments
        filedata = filedata.replace('{{!--', '<!--')
        filedata = filedata.replace('--}}', '-->')

        # Write the file out again
        with open(filename, 'w') as outfile:
          outfile.write(filedata)
