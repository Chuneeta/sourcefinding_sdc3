from argparse import ArgumentParser
import numpy as np
import pylab
from mwa_qa.read_image import Image
import os
import pandas as pd
from collections import OrderedDict
from mwa_qa import json_utils as ju

parser=ArgumentParser(description='generate source catalog')
parser.add_argument('fn', type=str, help='Name of input fitsfile')
parser.add_argument('--outfile', type=str, help='Name of output catalogue', default='src_catalogue')
args = parser.parse_args()

imagepath = '/Users/ridhima/Documents/SDC3/images/'
gaulCat = args.fn
data = np.loadtxt(gaulCat, dtype=str, delimiter=',')
nsources = data.shape[0]
ras = data[:, 4].astype(float)
eras = data[:, 5].astype(float)
decs = data[:, 6].astype(float)
edecs = data[:, 7].astype(float)
pfluxs_isl = data[:, 10].astype(float)
inds = np.argsort(pfluxs_isl)[::-1]
ras_s = ras[inds]
decs_s = decs[inds]

nfreqs = 200
freqs = np.linspace(106, 196.1 , nfreqs)
tfluxs = np.zeros((nfreqs))
pfluxs = np.zeros((nfreqs))
errs = np.zeros((nfreqs))
src_dict = OrderedDict()
for i in range(nsources):
	srcname = '({:.2f}, {:.2f})'.format(ras_s[i], decs_s[i])
	src_dict[srcname] = OrderedDict()
	for j in range(nfreqs):
		imagename = os.path.join(imagepath, 'ZW2.msn_image_f{:.1f}MHz.fits'.format(freqs[j]))
		img = Image(imagename)
		srcflux = img.src_flux((ras_s[i], decs_s[i]))
		pfluxs[j] = srcflux[0]
		tfluxs[j] = srcflux[1]
		errs[j] = srcflux[2]	
	src_dict[srcname]['pflux'] = pfluxs
	src_dict[srcname]['tflux'] = tfluxs
	src_dict[srcname]['err'] = errs


ju.write_metrics(src_dict, args.outfile)


# plotting of ras and decs
#fig = pylab.figure()
#pylab.scatter(ras_s, decs_s, marker='o', c=pfluxs)
#pylab.xlabel('Right Ascension')
#pylab.ylabel('Declination')
#pylab.show()
