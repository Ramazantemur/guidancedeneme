function [los_vec, los_rate, los_accel, filt] = seeker_ftd(z_meas, filt, dt, params)
% SEEKER_FTD  Fixed-Time Exact Differentiator.
%
%   Extends the Super-Twisting Algorithm with an additional nonlinear term
%   (higher-order power law) that guarantees convergence within a
%   PREDEFINED TIME T_max, REGARDLESS of the initial estimation error.
%
%   This is the key advantage over the standard STA (finite-time):
%     STA : convergence time T ≤ f(x0) (depends on initial error)
%     FTD : convergence time T ≤ T_max (independent of initial error!)
%
%   Algorithm (per LOS angle axis):
%     e  = z0 - y_meas               [estimation error]
%     dz0/dt = z1 - k1*|e|^r1*sign(e) - k2*|e|^r2*sign(e)
%     dz1/dt =    - k3*sign(e)       - k4*|e|*sign(e)
%
%   where r1 < 1 (finite-time term) and r2 > 1 (fixed-time term).
%   The fixed-time convergence bound: T_max ≈ π/(2*sqrt(k3)) + 1/(k1*(1-r1))
%
%   Gain design for T_max seconds:
%     k3 = (π/(2*T_max))^2           [from fixed-time theory]
%     k4 = 2.0 (typically)
%     k1 = 1.5*sqrt(k3)              [STA-matching)
%     k2 = k1 * 0.5                  [fixed-time extra term]
%     r1 = 0.5 (standard STA exponent)
%     r2 = 1.5 (fixed-time exponent)
%
%   References:
%     Zimenko, K. et al. (2020). "Generalization of Supertwisting Algorithm
%       with Predefined Convergence Time." IFAC-PapersOnLine, 53(2),
%       pp. 6026–6031.
%     Cruz-Zavala, E. & Moreno, J.A. (2021). "Homogeneous High Order Sliding
%       Mode Design: A Lyapunov Approach." Automatica, 129, 109640.
%     Seeber, R. et al. (2021). "Saturated Super-Twisting Algorithm
%       with Predefined Convergence Time." IEEE CDC 2021.
%
%   params fields:
%     .ftd_Tmax    - predefined convergence time (s) — e.g. 2.0 s
%     .ftd_n_sub   - integration sub-steps (default 20)

    % ---- gains from predefined convergence time ---------------------
    T_max = params.ftd_Tmax;
    k3    = (pi / (2 * T_max))^2;    % fixed-time pole
    k4    = 2.0 * k3;                % damping of fixed-time term
    k1    = 1.5 * sqrt(k3);          % STA magnitude gain
    k2    = k1 * 0.5;                % fixed-time extra magnitude gain

    r1    = 0.5;     % finite-time exponent  (< 1)
    r2    = 1.5;     % fixed-time exponent   (> 1)

    % ---- angle measurement ------------------------------------------
    [az_meas, el_meas] = los_to_angles(z_meas);

    % ---- initialise -------------------------------------------------
    if isempty(filt.z_el)
        filt.z_el    = [el_meas; 0];
        filt.z_az    = [az_meas; 0];
        filt.z1_el_sm = 0;
        filt.z1_az_sm = 0;
    end

    z1_el_prev_sm = filt.z1_el_sm;
    z1_az_prev_sm = filt.z1_az_sm;

    % ---- integrate FTD (sub-steps for numerical stability) ----------
    n_sub = max(1, round(params.ftd_n_sub));
    h     = dt / n_sub;

    for i_sub = 1:n_sub   %#ok<FXUP>
        % --- Elevation ---
        e_el   = filt.z_el(1) - el_meas;
        % Two power-law correction terms:
        phi_el = k1 * sig_pow(e_el, r1) + k2 * sig_pow(e_el, r2);
        psi_el = k3 * sign_val(e_el)    + k4 * abs(e_el) * sign_val(e_el);

        dz0_el = -phi_el + filt.z_el(2);
        dz1_el = -psi_el;

        % --- Azimuth ---
        e_az   = angle_wrap(filt.z_az(1) - az_meas);
        phi_az = k1 * sig_pow(e_az, r1) + k2 * sig_pow(e_az, r2);
        psi_az = k3 * sign_val(e_az)    + k4 * abs(e_az) * sign_val(e_az);

        dz0_az = -phi_az + filt.z_az(2);
        dz1_az = -psi_az;

        filt.z_el = filt.z_el + h * [dz0_el; dz1_el];
        filt.z_az = filt.z_az + h * [dz0_az; dz1_az];
    end

    % ---- post-LP smoother on rate (suppress chattering) -------------
    tau_sm = 0.06;
    alpha  = tau_sm / (tau_sm + dt);
    filt.z1_el_sm = alpha * filt.z1_el_sm + (1-alpha) * filt.z_el(2);
    filt.z1_az_sm = alpha * filt.z1_az_sm + (1-alpha) * filt.z_az(2);

    % ---- acceleration via finite difference on smoothed rate --------
    el_accel = (filt.z1_el_sm - z1_el_prev_sm) / dt;
    az_accel = (filt.z1_az_sm - z1_az_prev_sm) / dt;

    % ---- outputs ----------------------------------------------------
    los_vec   = angles_to_los(filt.z_az(1), filt.z_el(1));
    los_rate  = los_rate_from_angles(filt.z_az(1), filt.z_el(1), ...
                                      filt.z1_az_sm, filt.z1_el_sm);
    los_accel = los_rate_from_angles(filt.z_az(1), filt.z_el(1), ...
                                      az_accel,       el_accel);
end

% ---- sig_pow: signed power function ---------------------------------
function y = sig_pow(x, p)
    % sig^p(x) = |x|^p * sign(x)
    y = (abs(x) + 1e-15)^p * sign_val(x);
end
function y = sign_val(x)
    if x > 0, y = 1; elseif x < 0, y = -1; else, y = 0; end
end
function [az, el] = los_to_angles(v)
    v = v/(norm(v)+eps);
    el = asin(max(-1,min(1,-v(3))));
    az = atan2(v(2),v(1));
end
function v = angles_to_los(az, el)
    v = [cos(el)*cos(az); cos(el)*sin(az); -sin(el)];
    v = v/(norm(v)+eps);
end
function omega = los_rate_from_angles(az, el, az_dot, el_dot)
    dv_del = [-sin(el)*cos(az); -sin(el)*sin(az); -cos(el)];
    dv_daz = [-cos(el)*sin(az);  cos(el)*cos(az);        0];
    omega  = el_dot * dv_del + az_dot * dv_daz;
end
function a = angle_wrap(a)
    while a > pi,  a = a - 2*pi; end
    while a < -pi, a = a + 2*pi; end
end
