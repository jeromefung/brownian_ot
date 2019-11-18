function [fx, fy, fz, tx, ty, tz] = ott_calc_force(pos, rot_matrix)
% Calculate optical forces and torques given beam-shape coefficients
% and T matrix of particle.
% Inputs:
%    x, y, z : coordinates of particle relative to beam 
%    r11, ..., r33 : elements of rotation matrix describing particle 
%        orientation. (Columns describe orientation of particle
%        axes in lab frame.)
global beam_obj;
global Tmatrix;
pos = transpose(pos);
%rot_matrix = [r11, r12, r13 ;
%              r21, r22, r23 ;
%              r31, r32, r33];
[fx, fy, fz, tx, ty, tz] = ott.forcetorque(beam_obj, Tmatrix, ...
    'position', pos, 'rotation', rot_matrix);