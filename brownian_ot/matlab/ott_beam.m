
function beam_nmax = ott_beam(lambda_0, pol_x, pol_y, NA, n_med)
% Create an ott beam object for a Gaussian beam via point matching.
% Return the maximum order beam_nmax of the beam's VSWF expansion.
% ott is unit-agnostic, but all units need to be consistent.
% Inputs:
%    lambda_0 : incident vacuum wavelength
%    pol_x : x component of polarization 
%    pol_y : y component of polarization
%    NA : numerical aperture of focusing objective
%    n_med : medium refractive index
% Use a global variable b/c of calls from matlab engine
global beam_obj;
pol = [pol_x, pol_y]; % Matlab engine can't pass numpy arrays
beam_obj = ott.BscPmGauss('NA', NA, 'polarisation', double(pol), ...
    'index_medium', n_med, 'wavelength0', lambda_0);
beam_obj.power = 1.0; % Normalize the beam power
beam_nmax = beam_obj.Nmax ;