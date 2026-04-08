function [los_vec, los_rate, los_accel, filt] = seeker_std(z_meas, filt, dt, params)
% SEEKER_STD  Super-Twisting Algorithm (STA) exact differentiator.
%
%   Implements the 1st-order Super-Twisting Algorithm (Moreno & Osorio 2012)
%   for robust estimation of LOS angle and LOS angular rate.
%   Acceleration is estimated by finite difference on the rate estimate.
%
%   For each LOS angle measurement y(t), the continuous-time STA is:
%
%     dz0/dt = -lambda1 * |z0 - y|^(1/2) * sign(z0 - y) + z1   [angle]
%      dz1/dt = -lambda2 * sign(z0 - y)                          [rate]
%
%   Gain design: given signal bound L (|d²y/dt²| <= L):
%     lambda1 = 1.5 * sqrt(L)
%     lambda2 = 1.1 * L
%
%   This 1st-order STA provides:
%     z0 → filtered angle (converges in finite time to y)
%     z1 → estimated angular rate (converges in finite time)
%   For acceleration: finite-difference on z1.
%
%   References:
%     Moreno & Osorio (2012), "Strict Lyapunov Functions for the
%       Super-Twisting Algorithm", IEEE TAC 57(4), pp. 1035-1040
%     Levant (1998), "Robust Exact Differentiation via Sliding Mode
%       Technique", Automatica 34(3), pp. 379-384
%
%   params fields:
%     .std_L     - Lipschitz constant on 2nd derivative of LOS angle (rad/s^2)
%     .std_n_sub - number of sub-steps per main dt (default 10)

    % ---- convert to spherical angles ---------------------------------
    [az_meas, el_meas] = los_to_angles(z_meas);

    % ---- gains from Lipschitz constant L ----------------------------
    L  = params.std_L;
    l1 = 1.5 * sqrt(L);
    l2 = 1.1 * L;

    % ---- initialise (first call: set angle with zero rate) ----------
    if isempty(filt.z_el)
        filt.z_el       = [el_meas; 0];   % [angle; rate] raw STA states
        filt.z_az       = [az_meas; 0];
        filt.z1_el_sm   = 0;              % smoothed rate (post-LP)
        filt.z1_az_sm   = 0;
    end

    % ---- save previous smooth rate for acceleration FD ---------------
    z1_el_prev_sm = filt.z1_el_sm;
    z1_az_prev_sm = filt.z1_az_sm;

    % ---- integrate with sub-steps for numerical stability -----------
    n_sub = max(1, round(params.std_n_sub));
    h     = dt / n_sub;

    for i_sub = 1:n_sub
        % --- Elevation axis ---
        e_el   = filt.z_el(1) - el_meas;
        dz0_el = -l1 * abs_pow(e_el, 0.5) + filt.z_el(2);
        dz1_el = -l2 * sign_val(e_el);

        % --- Azimuth axis ---
        e_az   = angle_wrap(filt.z_az(1) - az_meas);
        dz0_az = -l1 * abs_pow(e_az, 0.5) + filt.z_az(2);
        dz1_az = -l2 * sign_val(e_az);

        filt.z_el = filt.z_el + h * [dz0_el; dz1_el];
        filt.z_az = filt.z_az + h * [dz0_az; dz1_az];
    end

    % ---- post-LP smoother on rate to suppress chattering ------------
    tau_sm = 0.08;                        % smoothing time constant (s)
    alpha  = tau_sm / (tau_sm + dt);
    filt.z1_el_sm = alpha * filt.z1_el_sm + (1-alpha) * filt.z_el(2);
    filt.z1_az_sm = alpha * filt.z1_az_sm + (1-alpha) * filt.z_az(2);

    % ---- acceleration by finite difference on smoothed rate ---------
    el_accel = (filt.z1_el_sm - z1_el_prev_sm) / dt;
    az_accel = (filt.z1_az_sm - z1_az_prev_sm) / dt;

    % ---- convert to NED Cartesian -----------------------------------
    los_vec   = angles_to_los(filt.z_az(1), filt.z_el(1));
    los_rate  = los_rate_from_angles(filt.z_az(1), filt.z_el(1), ...
                                      filt.z_az(2), filt.z_el(2));
    los_accel = los_rate_from_angles(filt.z_az(1), filt.z_el(1), ...
                                      az_accel, el_accel);
end

% ---- local helpers --------------------------------------------------
function y = abs_pow(x, p)
    y = (abs(x) + 1e-14)^p;  % epsilon avoids exact 0 issues
end
function y = sign_val(x)
    if x > 0, y = 1; elseif x < 0, y = -1; else, y = 0; end
end
function [az, el] = los_to_angles(v)
    v = v / (norm(v) + eps);
    el = asin(max(-1, min(1, -v(3))));
    az = atan2(v(2), v(1));
end
function v = angles_to_los(az, el)
    v = [cos(el)*cos(az); cos(el)*sin(az); -sin(el)];
    v = v / (norm(v) + eps);
end
function omega = los_rate_from_angles(az, el, az_dot, el_dot)
    dv_del = [-sin(el)*cos(az); -sin(el)*sin(az); -cos(el)];
    dv_daz = [-cos(el)*sin(az);  cos(el)*cos(az);        0];
    omega  = el_dot * dv_del + az_dot * dv_daz;
end
function a = angle_wrap(a)
    while a >  pi, a = a - 2*pi; end
    while a < -pi, a = a + 2*pi; end
end
