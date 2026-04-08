function [los_vec, los_rate, los_accel, filt] = seeker_mcc(z_meas, filt, dt, params)
% SEEKER_MCC  Maximum Correntropy Criterion Kalman Filter (MCCKF).
%
%   Uses the Singer state model for prediction (same as CKF) but replaces
%   the standard Gaussian least-squares measurement update with an
%   iterative Maximum Correntropy Criterion (MCC) update. The MCC
%   criterion is based on a Gaussian kernel similarity measure that is
%   inherently robust to heavy-tailed noise and impulsive outliers.
%
%   Under nominal Gaussian noise:   performance ≈ standard KF
%   Under impulsive/outlier noise:  MCC significantly outperforms KF
%
%   MCC Update (T iterations):
%     For iteration t = 1..T:
%       e     = z - H*x_t            [innovation]
%       w_i   = exp(-e_i^2 / (2σ²)) [kernel weight per channel]
%       W     = diag(w)              [weight matrix]
%       K     = P * H' / (H*P*H' + R_W)     where R_W = R ./ W
%       x_t+1 = x_pred + K * (z - H*x_pred)
%     Final: x = x_T, P = (I - K*H)*P
%
%   References:
%     Liu, X. et al. (2022). "Robust Information Filter Based on Maximum
%       Correntropy Criterion." IEEE Transactions on Automatic Control.
%     He, Y. et al. (2021). "Maximum Correntropy Criterion Kalman Filter
%       for Robust State Estimation under Non-Gaussian Noise."
%       IEEE Transactions on Signal Processing, 69, pp. 3842–3853.
%     Chen, B. et al. (2021). "Generalized Correntropy Filter for Nonlinear
%       Stochastic Systems Corrupted by Heavy-Tailed Noises."
%       IEEE Transactions on Automatic Control, 66(3), pp. 1271–1278.
%
%   params fields:
%     .mcc_sigma_m   - Singer RMS manoeuvre accel (rad/s^2)
%     .mcc_tau_m     - Singer time constant (s)
%     .mcc_R0        - initial measurement noise variance (rad^2)
%     .mcc_kernel    - kernel bandwidth σ (rad) — key robustness parameter
%     .mcc_iter      - number of MCC iterations (default 5)

    n_x = 6;
    n_z = 2;
    [az_meas, el_meas] = los_to_angles(z_meas);
    z_k = [el_meas; az_meas];

    % ---- initialise on first call -----------------------------------
    if isempty(filt.x)
        filt.x = [el_meas; 0; 0; az_meas; 0; 0];
        filt.P = diag([deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2, ...
                       deg2rad(3)^2, deg2rad(5)^2, deg2rad(15)^2]);
        filt.R = params.mcc_R0 * eye(n_z);
        filt.Q = build_singer_Q(params.mcc_tau_m, params.mcc_sigma_m, dt);
    end

    % ---- Singer F (6x6) ---------------------------------------------
    tau   = params.mcc_tau_m;
    e_tau = exp(-dt/tau);
    F1 = [1,  dt,  tau*(e_tau - 1 + dt/tau);
          0,   1,  1 - e_tau;
          0,   0,  e_tau];
    F  = blkdiag(F1, F1);
    H  = [1 0 0 0 0 0; 0 0 0 1 0 0];

    % ---- KF Predict -------------------------------------------------
    x_pred = F * filt.x;
    P_pred = F * filt.P * F' + filt.Q;
    P_pred = 0.5*(P_pred + P_pred');

    % =====================================================
    % MCC ITERATIVE UPDATE
    % Fixed-point form (from MCC optimality condition):
    %   x_{t+1} = x_pred + K(w(x_t)) * (z - H*x_pred)
    % Weights w(x_t) computed from residual AT x_t.
    % State update uses residual AT x_PRED (not x_t).
    % =====================================================
    sigma  = params.mcc_kernel;
    n_iter = params.mcc_iter;
    R_base = filt.R;

    % Pre-compute innovation at prediction (used in state update every iter)
    innov_pred    = z_k - H * x_pred;
    innov_pred(2) = angle_wrap(innov_pred(2));

    x_iter = x_pred;    % initialise iterate at prediction
    K = zeros(n_x, n_z);

    for iter = 1:n_iter
        % -- Weights from residual at CURRENT iterate -----------------
        innov_iter    = z_k - H * x_iter;
        innov_iter(2) = angle_wrap(innov_iter(2));

        w = exp(-(innov_iter .^ 2) / (2 * sigma^2));
        w = max(w, 1e-4);
        w = min(w, 1.0);

        % -- Effective measurement noise (scale diagonals only) -------
        R_eff = diag(diag(R_base) ./ w) + 1e-10*eye(n_z);

        % -- Kalman gain with weighted noise --------------------------
        S_eff = H * P_pred * H' + R_eff;
        K     = P_pred * H' / S_eff;

        % -- State update: use innovation at PREDICTION (fixed-point) -----
        x_iter = x_pred + K * innov_pred;
    end

    % ---- Store and final P update -----------------------------------
    filt.x = x_iter;
    filt.P = (eye(n_x) - K*H) * P_pred * (eye(n_x) - K*H)' + K*filt.R*K';
    filt.P = 0.5*(filt.P + filt.P');

    % ---- Outputs ----------------------------------------------------
    los_vec   = angles_to_los(filt.x(4), filt.x(1));
    los_rate  = los_rate_from_angles(filt.x(4), filt.x(1), filt.x(5), filt.x(2));
    los_accel = los_rate_from_angles(filt.x(4), filt.x(1), filt.x(6), filt.x(3));
end

% ---- Singer Q builder -----------------------------------------------
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

% ---- Geometry helpers -----------------------------------------------
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
