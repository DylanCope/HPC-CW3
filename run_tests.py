from subprocess import check_output, STDOUT
import numpy as np
import sys
import os
import time
import math 
from itertools import chain, product
from random import shuffle

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

def run_test( wgs, dim ):

	# dictionary mapping line nums to data to format
	info = { 3    : ('ocl_wgs%d_%s' % (wgs, dim)), \
                 5    : ('%soutput_wgs%d_%s' % (folder, wgs, dim)), \
		 22   : (dim, dim, wgs)  }

	lines = []
	with open( 'ocl_submit_template' ) as f:
		for i, line in enumerate( f.readlines() ):
			l = line % info[ i + 1 ] if i + 1 in info else line
			lines.append( l )
		
	run_filename = '%socl_submit_wgs%i_%s' % ( folder, wgs, dim )

	if not os.path.exists( run_filename ):
		check_output( 'echo "" > %s' % run_filename, \
				stderr = STDOUT, shell = True )

	with open( run_filename, 'w+' ) as f:
		f.flush()
		f.writelines( lines )

	cmd = 'qsub %s' % run_filename
	output = check_output( cmd, stderr = STDOUT, shell = True )



work_group_sizes = [ 128 ]
dimens = [ '128x128', '128x256', '256x256', '1024x1024' ]
params = list(product( work_group_sizes, dimens ))
shuffle(params)

sleeptime = 0
burst = 4
burst_counter = 0
for wgs, dim in params:
	if burst_counter == burst:
		time.sleep( sleeptime )
		burst_counter = 0
	burst_counter += 1
	run_test( wgs, dim )


