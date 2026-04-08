function [los_vec, los_rate, filt] = seeker_riekf(los_noisy, filt, dt, params)
% SEEKER_RIEKF — Algorithm 8: Robust Iterated EKF (Huber M-estimator)
%
%   Optimization-based approach that replaces the standard Gaussian
%   measurement update with a robust M-estimator (Huber loss function),
%   solved iteratively via IRLS (Iteratively Reweighted Least Squares).
%
%   State space: same as Algorithm 2 (EKF-LOS):
%     x = [el; el_dot; az; az_dot; el_bias; az_bias]   (6-state)
%
%   Process model: same kinematic model as Algorithm 2.
%
%   Standard EKF minimises the quadratic cost (Gaussian assumption):
%     J(x) = ||x - x_p||^2_{P_p^{-1}} + ||z - Hx||^2_{R^{-1}}
%
%   RIEKF-Huber minimises the robust cost:
%     J(x) = ||x - x_p||^2_{P_p^{-1}} + Σ_j ρ_H(r_j / σ_j)
%   where σ_j = sqrt(S_jj), r_j = z_j - H_j x, and:
%     ρ_H(u) = u^2/2          if |u| ≤ δ  (quadratic: Gaussian-like)
%              δ|u| - δ^2/2  if |u|  > δ  (linear: outlier downweighted)
%
%   Huber influence function (derivative of ρ_H):
%     ψ_H(u) = u         if |u| ≤ δ
%            = δ·sign(u) if |u| > δ
%
%   IRLS implementation (per step):
%     x_0 = x_pred
%     for i = 1..N_iter:
%       r      = z - H*x_i
%       σ_j    = sqrt(S_jj)  [from S = H P_p H' + R, computed once]
%       w_j    = min(1, δ·σ_j / |r_j|)   [per component, j=1..n_z]
%       W      = diag(w)
%       K      = P_p H' (H P_p H' + W^{-1} R)^{-1}   [standard KF gain with W]
%       x_{i+1} = x_p + K · ψ_H(r)   where ψ_H applied element-wise
%     P_up = (I - K·W·H) P_p
%
%   Why this beats standard EKF when noise is impulsive:
%     - For normal measurements (|r_j| ≤ δ σ_j): w_j = 1, same as EKF
%     - For outlier measurements (|r_j|  > δ σ_j): w_j < 1, downweighted
%     - Result: spikes are automatically detected and rejected per-component
%
%   params:
%     .riekf_R0       - measurement noise var (rad^2)
%     .riekf_q_rate   - rate process noise PSD
%     .riekf_q_bias   - bias process noise PSD
%     .riekf_delta    - Huber threshold (in units of innovation sigma)
%                       typical: 1.5 (95% efficiency for Gaussian)
%     .riekf_n_iter   - number of IRLS iterations per step (3-5 is enough)
%     .riekf_p0_ang, .riekf_p0_rate, .riekf_p0_bias  (initial P)

    n_x = 6;
    n_z = 2;

    % ---- Measurement: noisy LOS → angles ------------------------------------
    [az_m, el_m] = los_to_angles(los_noisy);
    z_k = [el_m; az_m];

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        filt.x = [el_m; 0; az_m; 0; 0; 0];
        filt.P = diag([params.riekf_p0_ang,  params.riekf_p0_rate, ...
                       params.riekf_p0_ang,  params.riekf_p0_rate, ...
                       params.riekf_p0_bias, params.riekf_p0_bias].^2);
        filt.R = params.riekf_R0 * eye(n_z);
    end

    % ---- State transition & process noise (same as Alg 2) -------------------
    F = [1, dt,  0,  0,  0,  0;
         0,  1,  0,  0,  0,  0;
         0,  0,  1, dt,  0,  0;
         0,  0,  0,  1,  0,  0;
         0,  0,  0,  0,  1,  0;
         0,  0,  0,  0,  0,  1];

    q_r = params.riekf_q_rate;
    q_b = params.riekf_q_bias;
    Q   = diag([q_r*dt^3/3, q_r*dt, q_r*dt^3/3, q_r*dt, q_b*dt, q_b*dt]);

    H = [1, 0, 0, 0, 1, 0;
         0, 0, 1, 0, 0, 1];

    % ---- PREDICT (standard EKF) ----------------------------------------------
    x_p = F * filt.x;
    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- Pre-compute innovation covariance S (once, used for scaling) --------
    S   = H * P_p * H' + filt.R;          % (2×2) standard innovation cov
    sig = sqrt(max(eps, diag(S)));         % per-component innovation sigma

    % ---- IRLS robust update --------------------------------------------------
    delta   = params.riekf_delta;          % Huber threshold [sigma units]
    N_iter  = params.riekf_n_iter;
    x_iter  = x_p;

    for iter = 1:N_iter
        r = z_k - H * x_iter;
        r(2) = angle_wrap(r(2));

        % Per-component Huber weights: w_j = min(1, delta*sigma_j / |r_j|)
        w = min(1, delta * sig ./ max(abs(r), 1e-12));
        W_diag = diag(w);                  % (2×2) diagonal weight matrix
        W_inv  = diag(1 ./ w);            % (2×2) inverse

        % Effective noise covariance with Huber weighting
        R_eff = W_inv * filt.R;            % higher R for outliers (downweight)

        % Kalman gain with effective noise covariance
        S_eff = H * P_p * H' + R_eff;
        K     = P_p * H' / S_eff;

        % Huber pseudo-residual: ψ_H applied element-wise
        psi = min(1, delta * sig ./ max(abs(r), 1e-12)) .* r;
        % (equivalent to: ψ_H(r_j/sig_j)*sig_j for each component)

        x_iter = x_p + K * psi;

        % Check convergence
        if iter > 1 && norm(x_iter - x_prev) < 1e-8
            break
        end
        x_prev = x_iter;
    end

    % ---- Final covariance update (using last K and W) -----------------------
    P_up = (eye(n_x) - K * W_diag * H) * P_p * (eye(n_x) - K * W_diag * H)' ...
           + K * W_diag * filt.R * (K * W_diag)';
    P_up = 0.5*(P_up + P_up');

    filt.x = x_iter;
    filt.P = P_up;

    % ---- Outputs ------------------------------------------------------------
    el_est     = filt.x(1);
    el_dot_est = filt.x(2);
    az_est     = filt.x(3);
    az_dot_est = filt.x(4);

    los_vec  = angles_to_los(az_est, el_est);
    los_rate = los_rate_from_angles(az_est, el_est, az_dot_est, el_dot_est);
end

% ===== geometry helpers =====================================================
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
    a = mod(a + pi, 2*pi) - pi;
end
