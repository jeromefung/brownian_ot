
function beam_nmax = ott_beam(lambda_0, pol_x, pol_y, NA, n_med, mode)
% Create an ott beam object for a Gaussian beam via point matching.
% Return the maximum order beam_nmax of the beam's VSWF expansion.
% ott is unit-agnostic, but all units need to be consistent.
% Inputs:
%    lambda_0 : incident vacuum wavelength
%    pol_x : x component of polarization
%    pol_y : y component of polarization
%    NA : numerical aperture of focusing objective
%    n_med : medium refractive index
%    mode : LG mode [radial, azimuthal]
% Use a global variable b/c of calls from matlab engine
global beam_obj;
pol = [double(pol_x), double(pol_y)]; % Matlab engine can't pass numpy arrays
  % Need to make polarization explicitly double complex.
  % Otherwise, if a jones vector like [1, 1j] is passed in
  % Matlab thinks it is dealing with a complex integer.
  % Also, there is surprising Matlab behavior if you create an array
  % with an integer and a float/complex: the entire array is cast as
  % integer. This behavior is documented:
  % https://www.mathworks.com/help/matlab/matlab_prog/combining-integer-and-noninteger-data.html
beam_obj = ott.BscPmGauss('type', 'lg', 'mode', mode, 'NA', NA, ...
                          'polarisation', pol, 'index_medium', n_med, ...
                          'wavelength0', lambda_0);
beam_obj.power = 1.0; % Normalize the beam power
beam_nmax = beam_obj.Nmax ;
%disp(beam_obj.polarisation);
