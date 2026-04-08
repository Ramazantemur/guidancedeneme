function [los_vec, los_rate, los_accel, filt] = seeker_alphabeta(z_meas, filt, dt, params)
% SEEKER_ALPHABETA  Alpha-Beta-Gamma fixed-gain LOS tracker.
%
%   Classic 3-state polynomial tracker (Benedict-Bordner filter).
%   Estimates LOS angle, rate, and acceleration per axis.
%
%   State per axis: x = [angle; rate; accel]
%
%   Gains derived from bandwidth wn and critical damping (zeta=1):
%     alpha = 1 - exp(-wn*dt)^2   (angle gain)
%     beta  = 2*(1-exp(-wn*dt)) - alpha (rate gain)
%     gamma = 0.5*beta^2/alpha    (accel gain)
%
%   Reference: Benedict & Bordner (1962); Bar-Shalom "Estimation with
%              Applications to Tracking and Navigation" Ch.3
%
%   INTERFACE (same for all seeker filters):
%     z_meas  : 3x1 raw noisy unit LOS vector
%     filt    : filter state struct
%     dt      : timestep (s)
%     params  : struct with .abg_wn (bandwidth rad/s)
%   RETURNS:
%     los_vec   : 3x1 filtered unit LOS vector
%     los_rate  : 3x1 estimated LOS angular rate (rad/s)
%     los_accel : 3x1 estimated LOS angular acceleration (rad/s^2)
%     filt      : updated filter struct

    % ---- convert unit LOS vector to spherical angles (elevation, azimuth)
    [az_meas, el_meas] = los_to_angles(z_meas);

    % ---- initialise on first call -------------------------------------
    if isempty(filt.x_el)
        filt.x_el = [el_meas; 0; 0];
        filt.x_az = [az_meas; 0; 0];
    end

    % ---- compute gains -----------------------------------------------
    wn    = params.abg_wn;
    a     = exp(-wn * dt);
    alpha = 1 - a^2;
    beta  = 2*(1 - a) - alpha;
    gamma = 0.5 * beta^2 / (alpha + eps);

    % ---- predict step ------------------------------------------------
    F = [1  dt  0.5*dt^2;
         0   1       dt;
         0   0        1];

    x_el_pred = F * filt.x_el;
    x_az_pred = F * filt.x_az;

    % ---- measurement update ------------------------------------------
    res_el = el_meas - x_el_pred(1);
    res_az = angle_wrap(az_meas - x_az_pred(1));

    K = [alpha; beta/dt; gamma/dt^2];

    filt.x_el = x_el_pred + K * res_el;
    filt.x_az = x_az_pred + K * res_az;

    % ---- convert back to 3D ------------------------------------------
    los_vec   = angles_to_los(filt.x_az(1), filt.x_el(1));
    los_rate  = los_rate_from_angles(filt.x_az(1), filt.x_el(1), ...
                                      filt.x_az(2), filt.x_el(2));
    los_accel = los_rate_from_angles(filt.x_az(1), filt.x_el(1), ...
                                      filt.x_az(3), filt.x_el(3));
end

% =========================================================================
% SHARED UTILITIES (used by all seeker_*.m files — duplicated here for
%                   standalone operation; factor into a helper later)
% =========================================================================
function [az, el] = los_to_angles(v)
    v = v / (norm(v) + eps);
    el = asin(max(-1, min(1, -v(3))));   % NED: z down, +el = up
    az = atan2(v(2), v(1));              % azimuth from North
end

function v = angles_to_los(az, el)
    v = [cos(el)*cos(az); cos(el)*sin(az); -sin(el)];
    v = v / (norm(v) + eps);
end

function omega = los_rate_from_angles(az, el, az_dot, el_dot)
    % Convert spherical angle rates to Cartesian LOS rate vector
    % omega = d(los_hat)/dt expressed in NED
    dv_del = [-sin(el)*cos(az); -sin(el)*sin(az); -cos(el)];
    dv_daz = [-cos(el)*sin(az);  cos(el)*cos(az);        0];
    omega  = el_dot * dv_del + az_dot * dv_daz;
end

function a = angle_wrap(a)
    while a >  pi, a = a - 2*pi; end
    while a < -pi, a = a + 2*pi; end
end
