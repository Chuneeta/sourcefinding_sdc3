from argparse import ArgumentParser
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.wcs import WCS
import erfa
from astropy.coordinates import SkyCoord
from astropy.modeling.functional_models import AiryDisk2D
import numpy as np
import pylab

latitude = -26.82472208
longitude = 116.7644482
height = 0.0
FWHM_FACTOR = 2.35482004503
D2R = np.pi / 180.0
R2D = 180.0 / np.pi
ra0 = 0.0*D2R
dec0 = -30.0*D2R

def twoD_Gaussian(x : float,  y : float,
            xo : float,  yo : float,
            sigma_x : float,  sigma_y : float,
            cos_theta : float,  sin_theta : float,
            sin_2theta : float):
    """Explicitly a 2D gaussian function"""

    a = (cos_theta*cos_theta)/(2*sigma_x*sigma_x) + (sin_theta*sin_theta)/(2*sigma_y*sigma_y);
    b = -sin_2theta / (4*sigma_x*sigma_x) + sin_2theta / (4*sigma_y*sigma_y);
    c = (sin_theta*sin_theta)/(2*sigma_x*sigma_x) + (cos_theta*cos_theta)/(2*sigma_y*sigma_y);
    beam_real = np.exp( -( a*(x-xo)*(x-xo) + 2*b*(x-xo)*(y-yo) + c*(y-yo)*(y-yo) ));
    # beam_imag = 0.0;

    return beam_real


def rotate_around_sphere(rot_angle, az, za):
    """ignore this"""

    x = np.sin( za ) * np.cos( az )
    y = np.sin( za ) * np.sin( az )
    z = np.cos( za )

    # ##rotate about y
    x_rot = np.cos(rot_angle)*x + np.sin(rot_angle)*z
    y_rot = y
    z_rot = -np.sin(rot_angle)*x + np.cos(rot_angle)*z

    az_new = np.arctan2(y_rot, x_rot)
    za_new = np.arccos(z_rot / np.sqrt(x_rot**2 + y_rot**2 + z_rot**2 ))

    return az_new, za_new

def get_lmn(ra, ra0, dec, dec0):
    '''Calculate l,m, for a given phase centre ra0,dec0 and sky point ra,dec
    Enter angles in radians'''

    ##RTS way of doing it
    cdec0 = np.cos(dec0)
    sdec0 = np.sin(dec0)
    cdec = np.cos(dec)
    sdec = np.sin(dec)
    cdra = np.cos(ra-ra0)
    sdra = np.sin(ra-ra0)
    l = cdec*sdra
    m = sdec*cdec0 - cdec*sdec0*cdra
    n = sdec*sdec0 + cdec*cdec0*cdra

    return l,m,n

def generate_beam_coords(azs, els, LST, latitude, ra0, dec0):
    """This convert the az, els back into ra, dec, and then into l,m
    coords that seem to get the sky curvature correct

    Parameters
    ----------
    azs : np.ndarray
        Azmiuths to calculate the beam toward
    els : np.ndarray
        Elevations to calculate the beam toward
    LST : float
        Local sidereal time for this time step
    latitude : float
        The latitude of the array
    ra0 : float
        RA of the phase centre of the observation
    dec0 : float
        Dec of the phase centre of the observation

    Returns
    -------
    cent_l, cent_m, beam_ls, beam_ms
        cent_l - centre of beam coord in the l directinon
        cent_m - centre of beam coord in the m directinon
        beam_ls - l coords for the beam
        beam_ms - m coords for the beam
    """

    has, decs = erfa.ae2hd(azs, els, latitude)

    ras = LST - has

    beam_ls, beam_ms, _ = get_lmn(ras, LST, decs, latitude)
    cent_l, cent_m, n0 = get_lmn(ra0, LST, dec0, latitude)

    return cent_l, cent_m, beam_ls, beam_ms

def gauss_beam_azel(azs : np.ndarray, els : np.ndarray,
                    LST : float, fwhm : float, beam_ref_freq : float, freq : float,
                    ra0 : float, dec0 : float, latitude : float):
    """Calculate a phase-tracked Gaussian beam for a given set of azimuth, elevation
    coords, beam parameters, and phase centre, at a given LST and frequency
    
    All angles in radians

    Parameters
    ----------
    azs : np.ndarray
        Azmiuths to calculate the beam toward
    els : np.ndarray
        Elevations to calculate the beam toward
    LST : float
        Local sidereal time for this time step
    fwhm : float
        The full-width half-maximum of the beam at the reference frequency. Used
        to scale the beam width with frequency
    beam_ref_freq : float
        The reference frequency to the fwhm of the beam (Hz)
    freq : float
        The frequency to calculate the beam at
    ra0 : float
        RA of the phase centre of the observation
    dec0 : float
        Dec of the phase centre of the observation
    latitude : float
        The latitude of the array

    Returns
    -------
    np.ndarray
        Set of real values for a Stokes I beam
    """


    cent_l, cent_m, beam_ls, beam_ms = generate_beam_coords(azs, els, LST, latitude, ra0, dec0)

    ##these are related to position angle, which I'm setting to zero
    cos_theta = 1.0
    sin_theta = 0.0
    sin_2theta = 0.0

    ##scale fwhm to be in l,m coords 
    fwhm_lm = np.sin(fwhm)

    ##now convert from fwhm to std, and scale by frequency
    std = (fwhm_lm / FWHM_FACTOR) * (beam_ref_freq / freq)
	    ##make this a symmetric gaussian
    std_l = std
    std_m = std

    beam_real = twoD_Gaussian(beam_ls, beam_ms, cent_l, cent_m, std_l, std_m,
                              cos_theta, sin_theta, sin_2theta)

    return beam_real

def airy_beam_azel(azs : np.ndarray, els : np.ndarray,
                   LST : float, extrap_freq : float,
                   ra0 : float, dec0 : float, latitude : float):
    """Calculate a phase-tracked Airy-disk beam for a given set of azimuth, elevation
    coords, beam parameters, and phase centre, at a given LST and frequency
    
    All angles in radians

    Parameters
    ----------
    azs : np.ndarray
        Azmiuths to calculate the beam toward
    els : np.ndarray
        Elevations to calculate the beam toward
    LST : float
        Local sidereal time for this time step
    beam_ref_freq : float
        The reference frequency to the fwhm of the beam (Hz)
    freq : float
        The frequency to calculate the beam at
    ra0 : float
        RA of the phase centre of the observation
    dec0 : float
        Dec of the phase centre of the observation
    latitude : float
        The latitude of the array

    Returns
    -------
    np.ndarray
        Set of real values for a Stokes I beam
    """
    amplitude = 1

    ##hardcoded params that look good; TODO is to fit them properly
    beam_ref_freq = 106000000.0
    radius = 5.15*D2R*(beam_ref_freq / extrap_freq)

    cent_l, cent_m, beam_ls, beam_ms = generate_beam_coords(azs, els, LST, latitude, ra0, dec0)

    ##This function has returned a 2D array of airy disk valuyes
    airy_values = AiryDisk2D.evaluate(beam_ls, beam_ms, amplitude, cent_l, cent_m, radius)

    return airy_values


if __name__ == '__main__':
	parser=ArgumentParser(description='generate beam corrected image')
	parser.add_argument('fitsfiles', nargs='+', type=str, help='Name of input fitsfile')
	args = parser.parse_args()

	LST = 0.0
	for fn in args.fitsfiles:
		print ('Reading {}'.format(fn))
		with fits.open(fn) as hdu:
			wcs = WCS(hdu[0].header).celestial
			imdata = hdu[0].data
			num_x = hdu[0].header['NAXIS1']
			num_y = hdu[0].header['NAXIS2']
			freq = hdu[0].header['CRVAL3']

		coord_x, coord_y = np.meshgrid(range(num_x), range(num_y))

		ras, decs = wcs.all_pix2world(coord_x, coord_y, 0)
		ras *= D2R
		decs *= D2R
		has = LST*D2R - ras
		azs, els = erfa.hd2ae(has, decs, latitude*D2R)
		# generating airy beam
		airy_values = airy_beam_azel(azs, els, LST*D2R, freq, ra0, dec0, latitude*D2R)
		airy_beam = airy_values.reshape((1, 1, num_x, num_y)) 
		# primary beam correction
		pbeam_imdata = imdata / airy_beam 
		# writing out to fits file
		outfile = fn.replace('fits', '_pbeam.fits')
		fits.writeto(outfile, pbeam_imdata, header=hdu[0].header, overwrite=True)
		
		fig = pylab.figure(figsize=(10, 7))
		pylab.suptitle('{:.1f} MHz'.format(freq * 1e-6), size=13)
		pylab.subplot(221)
		pylab.imshow(np.abs(imdata[0, 0, :, :]), aspect='auto', vmin=-0.1, vmax=1.5)
		pylab.title('Data')
		pylab.colorbar()

		pylab.subplot(222)
		pylab.imshow(np.abs(airy_beam[0, 0, :, :]), aspect='auto', vmin=0, vmax=1)
		pylab.title('Beam')
		pylab.colorbar()

		pylab.subplot(223)
		pylab.imshow(np.abs(pbeam_imdata[0, 0, :, :]), aspect='auto', vmin=-0.1, vmax=1.5)
		pylab.title('Beam Corrected Data')
		pylab.colorbar()
		
		pylab.savefig(fn.replace('.fits', '_2dplots.png'), bbox_inches='tight')
		pylab.close()

		# cross cuts
		fig = pylab.figure(figsize=(8, 6))
		pylab.suptitle('{:.1f} MHz'.format(freq * 1e-6), size=13)
		pylab.subplot(121)
		pylab.semilogy(np.abs(pbeam_imdata[0, 0, 1024, :]))
		pylab.semilogy(np.abs(airy_beam[0, 0, 1024, :]), 'k--')		
		pylab.xlabel('xpix')
		pylab.ylabel('Pixel Value')
		pylab.grid(ls='dotted')
		pylab.ylim(10**-5, 10**1)

		pylab.subplot(122)
		pylab.semilogy(np.abs(pbeam_imdata[0, 0, :, 1024]))
		pylab.semilogy(np.abs(airy_beam[0, 0, :, 1024]), 'k--')
		pylab.xlabel('ypix')
		pylab.grid(ls='dotted')
		pylab.ylim(10**-5, 10**1)

		pylab.savefig(fn.replace('.fits', '_beamcross.png'), bbox_inches='tight')
		pylab.close()
