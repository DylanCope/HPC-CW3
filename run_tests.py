from subprocess import check_output, STDOUT
import numpy as np
import sys
import os
import time
import math 
from itertools import chain, product
from random import shuffle, random

try:
	folder = sys.argv[ 1 ]
except:
	testnum = 1 + len( os.listdir( './tests/' ) )
	folder = './tests/test%i/' % testnum

try:
	program = sys.argv[ 2 ]
except:
	program = './d2q9-bgk.exe'

if not os.path.exists( folder ):
	os.makedirs( folder )

print( sys.version )

def make_file( filename ):
	if not os.path.exists( filename ):
		check_output( 'echo "" > %s' % filename, \
				stderr = STDOUT, shell = True )

def gen_obstacles( dim, percentage ):
	dimstr = '%ix%i' % dim
	filename = '%sobs_%s_%f.dat' % (folder, dimstr, percentage)
	make_file( filename )
	r = lambda : 1 if random() > percentage else 0
	obstacles = [ (i, j, r()) for i, j in product(*map(range, dim)) ]
	with open( filename, 'w' ) as f:
		for data in filter( lambda x : x[2] == 1, obstacles ):
			f.write( '%d %d %d\n' % data )
	return filename

def run_test( wgs, dim, obsper = -1 ):

	dimstr = '%ix%i' % dim
	if 0 <= obsper <= 1:
		obstacles = gen_obstacles( dim, obsper )
		output_file = '%soutput_wgs%d_%s_%f' % (folder, wgs, dimstr, obsper)	
	else:
		obstacles = "obstacles_%s.dat" % dimstr
		output_file = '%soutput_wgs%d_%s' % (folder, wgs, dimstr)
	# dictionary mapping line nums to data to format
	info = { 3    : ('ocl_wgs%d_%s' % (wgs, dimstr)), \
                 5    : output_file, \
		 22   : (dimstr, obstacles, wgs)  }

	lines = []
	with open( 'ocl_submit_template' ) as f:
		for i, line in enumerate( f.readlines() ):
			l = line % info[ i + 1 ] if i + 1 in info else line
			lines.append( l )
		
	run_filename = '%socl_submit_wgs%i_%s' % ( folder, wgs, dimstr )
	make_file( run_filename )

	with open( run_filename, 'w+' ) as f:
		f.flush()
		f.writelines( lines )

	cmd = 'qsub %s' % run_filename
	output = check_output( cmd, stderr = STDOUT, shell = True )



work_group_sizes = [ 128 ]
obspers = [ 0.1, 0.5, 0.9 ]
dimens = [ (256, 256) ]#[ (128, 128), (128, 256), (256, 256), (1024, 1024) ]
params = list(product( work_group_sizes, dimens, obspers ))
shuffle(params)

sleeptime = 0
burst = 4
burst_counter = 0
for p in params:
	if burst_counter == burst:
		time.sleep( sleeptime )
		burst_counter = 0
	burst_counter += 1
	run_test( *p )


