###############################################################################
# Copyright (c) 2014, Blake Wulfe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
File: mergevec.py
Author: blake.w.wulfe@gmail.com
Date: 6/13/2014
File Description:

	This file contains a function that merges .vec files called "merge_vec_files".
	I made it as a replacement for mergevec.cpp (created by Naotoshi Seo.
	See: http://note.sonots.com/SciSoftware/haartraining/mergevec.cpp.html)
	in order to avoid recompiling openCV with mergevec.cpp.

	To use the function:
	(1) Place all .vec files to be merged in a single directory (vec_directory).
	(2) Navigate to this file in your CLI (terminal or cmd) and type "python mergevec.py -v your_vec_directory -o your_output_filename".

		The first argument (-v) is the name of the directory containing the .vec files
		The second argument (-o) is the name of the output file

	To test the output of the function:
	(1) Install openCV.
	(2) Navigate to the output file in your CLI (terminal or cmd).
	(2) Type "opencv_createsamples -w img_width -h img_height -vec output_filename".
		This should show the .vec files in sequence.

"""

import sys
import glob
import struct
import argparse
import traceback


def exception_response(e):
	exc_type, exc_value, exc_traceback = sys.exc_info()
	lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
	for line in lines:
		print(line)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', dest='vec_directory')
	parser.add_argument('-o', dest='output_filename')
	args = parser.parse_args()
	return (args.vec_directory, args.output_filename)

def merge_vec_files(vec_directory, output_vec_file):
	"""
	Iterates throught the .vec files in a directory and combines them.

	(1) Iterates through files getting a count of the total images in the .vec files
	(2) checks that the image sizes in all files are the same

	The format of a .vec file is:

	4 bytes denoting number of total images (int)
	4 bytes denoting size of images (int)
	2 bytes denoting min value (short)
	2 bytes denoting max value (short)

	ex: 	6400 0000 4605 0000 0000 0000

		hex		6400 0000  	4605 0000 		0000 		0000
			   	# images  	size of h * w		min		max
		dec	    	100     	1350			0 		0

	:type vec_directory: string
	:param vec_directory: Name of the directory containing .vec files to be combined.
				Do not end with slash. Ex: '/Users/username/Documents/vec_files'

	:type output_vec_file: string
	:param output_vec_file: Name of aggregate .vec file for output.
		Ex: '/Users/username/Documents/aggregate_vec_file.vec'

	"""

	# Check that the .vec directory does not end in '/' and if it does, remove it.
	if vec_directory.endswith('/'):
		vec_directory = vec_directory[:-1]
	# Get .vec files
	files = glob.glob('{0}/*.vec'.format(vec_directory))

	# Check to make sure there are .vec files in the directory
	if len(files) <= 0:
		print('Vec files to be mereged could not be found from directory: {0}'.format(vec_directory))
		sys.exit(1)
	# Check to make sure there are more than one .vec files
	if len(files) == 1:
		print('Only 1 vec file was found in directory: {0}. Cannot merge a single file.'.format(vec_directory))
		sys.exit(1)


	# Get the value for the first image size
	prev_image_size = 0
	try:
		with open(files[0], 'rb') as vecfile:
			content = b''.join((line) for line in vecfile.readlines())
			val = struct.unpack('<iihh', content[:12])
			prev_image_size = val[1]
	except IOError as e:
		print('An IO error occured while processing the file: {0}'.format(f))
		exception_response(e)


	# Get the total number of images
	total_num_images = 0
	for f in files:
		try:
			with open(f, 'rb') as vecfile:
				content = b''.join((line) for line in vecfile.readlines())
				val = struct.unpack('<iihh', content[:12])
				num_images = val[0]
				image_size = val[1]
				if image_size != prev_image_size:
					err_msg = """The image sizes in the .vec files differ. These values must be the same. \n The image size of file {0}: {1}\n
						The image size of previous files: {0}""".format(f, image_size, prev_image_size)
					sys.exit(err_msg)

				total_num_images += num_images
		except IOError as e:
			print('An IO error occured while processing the file: {0}'.format(f))
			exception_response(e)


	# Iterate through the .vec files, writing their data (not the header) to the output file
	# '<iihh' means 'little endian, int, int, short, short'
	header = struct.pack('<iihh', total_num_images, image_size, 0, 0)
	try:
		with open(output_vec_file, 'wb') as outputfile:
			outputfile.write(header)

			for f in files:
				with open(f, 'rb') as vecfile:
					content = b''.join((line) for line in vecfile.readlines())
					outputfile.write(bytearray(content[12:]))
	except Exception as e:
		exception_response(e)


if __name__ == '__main__':
	vec_directory, output_filename = get_args()
	if not vec_directory:
		sys.exit('mergvec requires a directory of vec files. Call mergevec.py with -v /your_vec_directory')
	if not output_filename:
		sys.exit('mergevec requires an output filename. Call mergevec.py with -o your_output_filename')

	merge_vec_files(vec_directory, output_filename)
