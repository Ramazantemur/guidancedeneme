function [los_vec, los_rate, target_est, filt] = seeker_ukf_relpos(los_noisy, uav_pos, uav_vel, filt, dt, params)
% SEEKER_UKF_RELPOS — Algorithm 6: Unscented Kalman Filter (Relative Position)
%
%   Same state space as Algorithm 4 (EKF-RelPos):
%     x = [drx; dry; drz; dvx; dvy; dvz]   (6-state, NED)
%
%   KEY ADVANTAGE over Alg 4: The nonlinear measurement
%     h(x) = x(1:3) / ||x(1:3)||
%   is handled via the Unscented Transform instead of EKF linearisation.
%   This eliminates the first-order approximation error in H_dr, which is
%   significant when range uncertainty is large (early in flight).
%
%   UKF parameters (alpha=1, kappa=0, beta=2  →  lambda=0):
%     2n+1 = 13 sigma points spread at ±sqrt(n*P) columns from the mean
%     Wm_0 = 0,     Wc_0 = 2 (beta term, Gaussian kurtosis)
%     Wm_i = Wc_i = 1/12  for i=1..12
%
%   Process model: same linear F as Alg 4 (UKF = EKF for linear predict).
%   Measurement update: propagates all 13 sigma points through h(x),
%   computes weighted predicted measurement, innovation covariance Pyy,
%   cross-correlation Pxy, and Kalman gain K = Pxy / Pyy.
%
%   Also uses altitude pseudo-measurement (target on ground) and
%   altitude-based range initialisation from Alg 4.
%
%   params: .ekfukf_R0, .ekfukf_q_pos, .ekfukf_q_vel,
%           .ekfukf_p0_pos, .ekfukf_p0_vel

    n_x = 6;
    n_z = 3;

    % ---- UKF weights (alpha=1, kappa=0, beta=2, lambda=0) -------------------
    lam  = 0;                    % lambda
    Wm   = [0; repmat(1/(2*n_x), 2*n_x, 1)];   % mean weights (13×1)
    Wc   = [2; repmat(1/(2*n_x), 2*n_x, 1)];   % cov weights: Wc_0=beta=2

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        % Altitude-based range initialisation (same as Alg 4 fix)
        if los_noisy(3) > 0.05
            r_init = -uav_pos(3) / los_noisy(3);
            r_init = max(200, min(5000, r_init));
        else
            r_init = 1500;
        end
        filt.x  = [los_noisy * r_init; -uav_vel];
        filt.P  = diag([params.ekfukf_p0_pos, params.ekfukf_p0_pos, params.ekfukf_p0_pos, ...
                        params.ekfukf_p0_vel, params.ekfukf_p0_vel, params.ekfukf_p0_vel].^2);
        filt.R  = params.ekfukf_R0 * eye(n_z);
        filt.vel_prev = uav_vel;
        filt.R_alt    = 5.0^2;
    end

    % ---- UAV acceleration (finite difference) --------------------------------
    if ~isempty(filt.vel_prev)
        a_uav = (uav_vel - filt.vel_prev) / dt;
    else
        a_uav = zeros(3,1);
    end
    filt.vel_prev = uav_vel;

    % ---- PREDICT (linear → same as EKF) -------------------------------------
    F = [eye(3), dt*eye(3); zeros(3), eye(3)];
    u = [zeros(3,1); -dt*a_uav];

    x_p = F * filt.x + u;

    q_p = params.ekfukf_q_pos;
    q_v = params.ekfukf_q_vel;
    Q   = diag([q_p*dt, q_p*dt, q_p*dt, q_v*dt, q_v*dt, q_v*dt]);

    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- Generate sigma points from (x_p, P_p) -----------------------------
    S_chol = chol((n_x + lam) * P_p, 'lower');   % lower-triangular sqrt
    % lam=0 → sqrt(n_x * P_p): spread matches covariance exactly
    Chi = zeros(n_x, 2*n_x + 1);
    Chi(:,1) = x_p;
    for i = 1:n_x
        Chi(:, 1+i)     = x_p + S_chol(:,i);
        Chi(:, 1+n_x+i) = x_p - S_chol(:,i);
    end

    % ---- Propagate sigma points through nonlinear measurement h(x) ----------
    Z_sig = zeros(n_z, 2*n_x+1);
    for i = 1:(2*n_x+1)
        dr_i = Chi(1:3, i);
        r_i  = norm(dr_i);
        if r_i < 0.5, r_i = 0.5; end
        Z_sig(:,i) = dr_i / r_i;
    end

    % ---- Predicted measurement & covariances --------------------------------
    z_hat = Z_sig * Wm;                  % weighted mean of sigma measurements
    z_hat = z_hat / max(norm(z_hat), 1e-6);   % re-normalise (unit LOS)

    Pyy = filt.R;                        % innovation covariance ← start with R
    Pxy = zeros(n_x, n_z);              % cross-correlation
    for i = 1:(2*n_x+1)
        dz = Z_sig(:,i) - z_hat;
        dx = Chi(:,i)   - x_p;
        Pyy = Pyy + Wc(i) * (dz * dz');
        Pxy = Pxy + Wc(i) * (dx * dz');
    end
    Pyy = 0.5*(Pyy + Pyy');

    % ---- Kalman gain & update -----------------------------------------------
    K     = Pxy / Pyy;
    innov = los_noisy - z_hat;

    x_up = x_p + K * innov;
    P_up = P_p - K * Pyy * K';
    P_up = 0.5*(P_up + P_up');

    % ---- Altitude pseudo-measurement (target on ground) ---------------------
    use_alt = ~isfield(params, 'ekfukf_use_alt') || params.ekfukf_use_alt;
    if use_alt
        H_alt = [0, 0, 1, 0, 0, 0];
        z_alt_pred = H_alt * x_up;
        z_alt_meas = -uav_pos(3);
        innov_alt  = z_alt_meas - z_alt_pred;
        S_alt = H_alt * P_up * H_alt' + filt.R_alt;
        K_alt = P_up * H_alt' / S_alt;
        x_up = x_up + K_alt * innov_alt;
        P_up = (eye(n_x) - K_alt * H_alt) * P_up;
        P_up = 0.5*(P_up + P_up');
    end

    filt.x = x_up;
    filt.P = P_up;

    % ---- Outputs ------------------------------------------------------------
    dr_est = filt.x(1:3);
    dv_est = filt.x(4:6);

    r_est = norm(dr_est);
    if r_est < 0.1, r_est = 0.1; end

    los_vec = dr_est / r_est;

    % LOS rate: dl/dt = (dv - (dv·l̂)l̂) / r
    l_hat   = los_vec;
    dv_perp = dv_est - dot(dv_est, l_hat) * l_hat;
    los_rate = dv_perp / r_est;

    target_est = uav_pos + dr_est;
end
