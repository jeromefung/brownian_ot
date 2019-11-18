function ott_tmatrix_sphere(n_p, r_p, lambda_0, n_med)
% Use ott's implementation of the Lorenz-Mie solution to calculate the
% T-matrix of a sphere.
% Inputs:
%     n_p : sphere refractive index
%     r_p : sphere radius (in same units as wavelength)
%     lambda_0 : vacuum wavelength
%     n_med : medium refractive index

global Tmatrix;

Tmatrix = ott.TmatrixMie.simple('sphere', r_p, 'wavelength0', lambda_0, ...
    'index_medium', n_med, 'index_particle', n_p);

