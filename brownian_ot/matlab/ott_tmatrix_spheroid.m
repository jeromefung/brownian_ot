function ott_tmatrix_spheroid(n_p, a, c, lambda_0, n_med)
% Use ott implementation of EBCM to calculate T-matrix for
% an (axisymmetric) spheroid. Spheroid has radius a in the x
% and y directions, and radius c in the z direction.
% Inputs:
%     n_p : sphere refractive index
%     a : spheroid x/y radius (in same units as wavelength)
%     c : spheroid z radius
%     lambda_0 : vacuum wavelength
%     n_med : medium refractive index

global Tmatrix;

Tmatrix = ott.TmatrixEbcm.simple('ellipsoid', [a, a, c], ...
    'wavelength0', lambda_0, 'index_medium', n_med, ...
				 'index_particle', n_p);
