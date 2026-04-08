function [los_vec, los_rate, los_accel, filt] = seeker_vbaekf(z_meas, filt, dt, params)
% SEEKER_VBAEKF  Variational Bayes Adaptive Extended Kalman Filter.
%
%   Simultaneously estimates LOS state AND measurement/process noise
%   covariances online using variational Bayes (VB) inference with an
%   inverse-Wishart prior on R and Q.
%
%   This makes the filter fully self-tuning: no need to specify noise
%   levels at design time. The filter adapts as the engagement evolves.
%
%   State: x = [el; el_dot; el_ddot; az; az_dot; az_ddot]   (6-state)
%   Model: Singer longitudinal/lateral acceleration model.
%
%   VB noise adaptation (Huang et al. 2017/2019):
%     R adaptation:
%       ρ_k = (ν_R + n_z) / (ν_R + k)          [forgetting weight]
%       R_k = (1-ρ_k)*R_{k-1} + ρ_k*(ε_k ε_k' - H P_{k-} H')
%     Q adaptation via inverse-Wishart posterior:
%       t_k = ν_Q / (ν_Q + 1)
%       Q_k = (1-t_k)*Q_{k-1} + t_k*(K_k ε_k ε_k' K_k')
%
%   References:
%     Huang et al. (2017), "A Novel Robust Gaussian-approximate
%       Fixed-interval Smoother", IEEE Signal Processing Letters
%     Huang et al. (2019), "Novel Adaptive Kalman Filter with
%       Variational Bayes for UAV Navigation", IEEE TAES 55(1)
%     Singer (1970), IEEE TAC 15(1)
%
%   params fields:
%     .vb_tau_m    - Singer manoeuvre time constant (s)
%     .vb_sigma_m  - Singer RMS manoeuvre accel (rad/s^2) [initial Q]
%     .vb_R0       - initial measurement noise variance (rad^2)
%     .vb_nu_R     - VB forgetting factor for R (default 10)
%     .vb_nu_Q     - VB forgetting factor for Q (default 5)

    n_x = 6;   % state dim
    n_z = 2;   % measurement dim (el, az)

    % ---- measurement ------------------------------------------------
    [az_meas, el_meas] = los_to_angles(z_meas);
    z_k = [el_meas; az_meas];

    % ---- initialise on first call -----------------------------------
    if isempty(filt.x)
        filt.x = [el_meas; 0; 0; az_meas; 0; 0];
        filt.P = diag([deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2, ...
                       deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2]);
        filt.R = params.vb_R0 * eye(n_z);
        filt.Q = build_singer_Q(params.vb_tau_m, params.vb_sigma_m, dt);
        filt.k = 1;
    end

    tau   = params.vb_tau_m;
    nu_R  = params.vb_nu_R;
    nu_Q  = params.vb_nu_Q;

    % ---- Singer F matrix (6x6 block diagonal) ----------------------
    e_tau = exp(-dt/tau);
    F1 = [1,  dt,  tau*(e_tau - 1 + dt/tau);
          0,   1,  1 - e_tau;
          0,   0,  e_tau];
    F  = blkdiag(F1, F1);

    H  = [1 0 0 0 0 0;
          0 0 0 1 0 0];

    % ---- EKF PREDICT ------------------------------------------------
    x_pred = F * filt.x;
    P_pred = F * filt.P * F' + filt.Q;
    P_pred = 0.5*(P_pred + P_pred');

    % ---- Innovation -------------------------------------------------
    z_pred   = H * x_pred;
    innov    = z_k - z_pred;
    innov(2) = angle_wrap(innov(2));

    % ---- EKF UPDATE -------------------------------------------------
    S   = H * P_pred * H' + filt.R;
    K   = P_pred * H' / S;
    x_up = x_pred + K * innov;
    P_up = (eye(n_x) - K*H) * P_pred * (eye(n_x) - K*H)' + K*filt.R*K';
    P_up = 0.5*(P_up + P_up');

    % =====================================================
    % VARIATIONAL BAYES NOISE COVARIANCE ADAPTATION
    % =====================================================
    k = filt.k;

    % --- Adapt R (measurement noise) ---
    rho_R   = (nu_R + n_z) / (nu_R + k);
    rho_R   = min(rho_R, 0.95);                 % clamp
    innov_outer = innov * innov';
    R_new = (1 - rho_R) * filt.R + ...
             rho_R * (innov_outer - H * P_pred * H');
    % Keep R symmetric and positive definite
    R_new = 0.5*(R_new + R_new');
    [~, p] = chol(R_new);
    if p > 0
        R_new = filt.R;    % reject bad update, keep previous
    end
    % Clamp R to physically reasonable range
    R_min = (deg2rad(0.01))^2;   % 0.01° minimum noise
    R_max = (deg2rad(5.0))^2;    % 5° maximum noise
    for ii = 1:n_z
        R_new(ii,ii) = max(R_min, min(R_max, R_new(ii,ii)));
    end

    % --- Adapt Q (process noise) via IW posterior ---
    t_Q = nu_Q / (nu_Q + 1);
    Q_innov = K * innov_outer * K';
    Q_new = (1 - t_Q) * filt.Q + t_Q * Q_innov;
    Q_new = 0.5*(Q_new + Q_new');
    [~, p] = chol(Q_new);
    if p > 0
        Q_new = filt.Q;
    end
    % Clamp Q diagonal
    for ii = 1:n_x
        Q_new(ii,ii) = max(1e-12, Q_new(ii,ii));
    end

    % ---- store updated state ----------------------------------------
    filt.x = x_up;
    filt.P = P_up;
    filt.R = R_new;
    filt.Q = Q_new;
    filt.k = k + 1;

    % ---- outputs ----------------------------------------------------
    los_vec   = angles_to_los(filt.x(4), filt.x(1));
    los_rate  = los_rate_from_angles(filt.x(4), filt.x(1), ...
                                      filt.x(5), filt.x(2));
    los_accel = los_rate_from_angles(filt.x(4), filt.x(1), ...
                                      filt.x(6), filt.x(3));
end

% ---- Singer Q matrix -----------------------------------------------
function Q = build_singer_Q(tau, sig_m, dt)
    e_tau = exp(-dt/tau);
    q33 = max(0, sig_m^2 * tau*(1 - e_tau^2));           % accel variance
    q22 = max(0, sig_m^2 * (dt - 2*tau*(1-e_tau) + tau*(1-e_tau^2)/2));
    q11 = max(0, sig_m^2 * dt^5 / 20);                   % angle variance (tiny)
    q23 = sig_m^2 * tau*(1 - e_tau)^2;
    q13 = sig_m^2 * (1 - e_tau)^2 / 2;
    q12 = sig_m^2 * ((1 + e_tau^2)/2 - e_tau) / (tau + eps);
    % Correct: [angle; rate; accel] → Q[1,1]=angle_noise, Q[3,3]=accel_noise
    Q1  = [q11, q12, q13;
           q12, q22, q23;
           q13, q23, q33];
    Q1  = 0.5*(Q1 + Q1');
    [~,p] = chol(Q1);
    if p > 0, Q1 = diag(max(diag(Q1), 1e-14)); end
    Q = blkdiag(Q1, Q1) + 1e-12*eye(6);
end

% ---- shared geometry helpers ----------------------------------------
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
