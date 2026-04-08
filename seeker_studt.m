function [los_vec, los_rate, los_accel, filt] = seeker_studt(z_meas, filt, dt, params)
% SEEKER_STUDT  Student-t Robust Kalman Filter (Gaussian Scale Mixture).
%
%   Replaces the standard Gaussian measurement likelihood p(z|x) with a
%   Student-t distribution p(z|x) ∝ (1 + z'/S*z/ν)^{-(ν+n)/2}.
%   Equivalent to a hierarchical model where each measurement is scaled
%   by a latent precision variable u ~ Gamma(ν/2, ν/2).
%
%   Small ν (e.g. 3-5) → heavy tails → robust to large outliers.
%   ν → ∞ → reduces to standard KF.
%
%   Algorithm (Variational Bayes E-M):
%     Predict: x_pred = F*x;  P_pred = F*P*F' + Q
%     For iter = 1..T:
%       u_k = (ν + n_z) / (ν + e_k' * S_k^{-1} * e_k)   [E-step: precision]
%       R_eff = R / u_k                                    [scaled noise]
%       K     = P * H' / (H*P*H' + R_eff)                 [Kalman gain]
%       x_iter = x_pred + K * (z - H*x_pred)              [M-step: mean]
%
%   References:
%     Huang, Y. et al. (2020). "Robust Kalman Filters Based on Gaussian Scale
%       Mixture Distributions with Application to Target Tracking."
%       IEEE Transactions on Systems, Man, and Cybernetics: Systems,
%       50(12), pp. 4842–4854.
%     Zhu, H. et al. (2021). "A Novel Robust Kalman Filter with Unknown
%       Non-Stationary Heavy-Tailed Noise." Automatica, 127, 109511.
%     Wang, G. et al. (2022). "Robust Gaussian Kalman Filter with
%       Outlier Detection." IEEE Transactions on Cybernetics.
%
%   params fields:
%     .st_tau_m    - Singer time constant (s)
%     .st_sigma_m  - Singer RMS accel (rad/s^2)
%     .st_R0       - initial measurement noise variance (rad^2)
%     .st_nu       - degrees of freedom (3-5 = heavy tail, >30 ≈ Gaussian)
%     .st_iter     - EM iterations per step (default 5)

    n_x = 6;
    n_z = 2;
    [az_meas, el_meas] = los_to_angles(z_meas);
    z_k = [el_meas; az_meas];

    % ---- initialise -------------------------------------------------
    if isempty(filt.x)
        filt.x = [el_meas; 0; 0; az_meas; 0; 0];
        filt.P = diag([deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2, ...
                       deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2]);
        filt.R = params.st_R0 * eye(n_z);
        filt.Q = build_singer_Q(params.st_tau_m, params.st_sigma_m, dt);
    end

    tau   = params.st_tau_m;
    e_tau = exp(-dt/tau);
    F1 = [1,  dt,  tau*(e_tau - 1 + dt/tau);
          0,   1,  1 - e_tau;
          0,   0,  e_tau];
    F  = blkdiag(F1, F1);
    H  = [1 0 0 0 0 0; 0 0 0 1 0 0];
    nu = params.st_nu;

    % ---- KF Predict -------------------------------------------------
    x_pred = F * filt.x;
    P_pred = F * filt.P * F' + filt.Q;
    P_pred = 0.5*(P_pred + P_pred');

    % =====================================================
    % STUDENT-T ITERATIVE E-M UPDATE
    % =====================================================
    n_iter = params.st_iter;
    R_base = filt.R;
    u      = 1.0;              % initial precision weight
    K      = zeros(n_x, n_z);

    for iter = 1:n_iter
        % -- M-step: compute Kalman gain with effective noise R/u ------
        R_eff = R_base / u;
        S_eff = H * P_pred * H' + R_eff;
        K     = P_pred * H' / S_eff;

        % -- Compute current state estimate ---------------------------
        innov    = z_k - H * x_pred;
        innov(2) = angle_wrap(innov(2));
        x_iter   = x_pred + K * innov;

        % -- E-step: update precision u from innovation ---------------
        % Normalized innovation squared (Mahalanobis distance)
        e_iter    = z_k - H * x_iter;
        e_iter(2) = angle_wrap(e_iter(2));
        S_cur     = H * P_pred * H' + R_base;
        delta2    = e_iter' / S_cur * e_iter;   % Mahalanobis^2

        % Student-t E-step: E[u] = (ν + n_z) / (ν + δ²)
        u = (nu + n_z) / (nu + delta2);
        u = max(u, 0.01);    % clamp: avoid R_eff → ∞
        u = min(u, 10.0);    % clamp: avoid R_eff → 0 (outlier overcorrection)
    end

    % ---- Final update -----------------------------------------------
    filt.x = x_pred + K * (z_k - H*x_pred - [0; 0]);
    innov_final    = z_k - H * x_pred;
    innov_final(2) = angle_wrap(innov_final(2));
    filt.x = x_pred + K * innov_final;
    filt.P = (eye(n_x) - K*H) * P_pred * (eye(n_x) - K*H)' + K*(R_base/u)*K';
    filt.P = 0.5*(filt.P + filt.P');

    % ---- Outputs ----------------------------------------------------
    los_vec   = angles_to_los(filt.x(4), filt.x(1));
    los_rate  = los_rate_from_angles(filt.x(4), filt.x(1), filt.x(5), filt.x(2));
    los_accel = los_rate_from_angles(filt.x(4), filt.x(1), filt.x(6), filt.x(3));
end

% ---- Helpers --------------------------------------------------------
function Q = build_singer_Q(tau, sig_m, dt)
    e_tau = exp(-dt/tau);
    q33 = max(0, sig_m^2 * tau*(1 - e_tau^2));
    q22 = max(0, sig_m^2 * (dt - 2*tau*(1-e_tau) + tau*(1-e_tau^2)/2));
    q11 = max(0, sig_m^2 * dt^5 / 20);
    q23 = sig_m^2 * tau*(1 - e_tau)^2;
    q13 = sig_m^2 * (1 - e_tau)^2 / 2;
    q12 = sig_m^2 * ((1 + e_tau^2)/2 - e_tau) / (tau + eps);
    Q1  = [q11, q12, q13; q12, q22, q23; q13, q23, q33];
    Q1  = 0.5*(Q1 + Q1');
    [~,p] = chol(Q1);
    if p > 0, Q1 = diag(max(diag(Q1), 1e-14)); end
    Q = blkdiag(Q1, Q1) + 1e-12*eye(6);
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
