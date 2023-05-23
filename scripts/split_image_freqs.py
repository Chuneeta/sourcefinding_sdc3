from argparse import ArgumentParser
from astropy.io import fits
import numpy as np

parser=ArgumentParser(description='generate beam corrected image')
parser.add_argument('fitsfile', type=str, help='Name of input fitsfile')
args = parser.parse_args()

hdu = fits.open(args.fitsfile)
hdr = hdu['PRIMARY'].header
data = hdu['PRIMARY'].data
freqs = np.linspace(106, 106 + 90.1, 901)
_sh = data.shape

for i in range(_sh[0]):
	outfile = args.fitsfile.replace('.fits', '_f{:.1f}MHz.fits'.format(freqs[i]))
	hdr['CRVAL3'] = freqs[i] * 1e6
	fits.writeto(outfile, data[i, :, :].reshape((1, 1, _sh[1], _sh[2])), hdr, overwrite=True)

