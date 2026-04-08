function [los_vec, los_rate, los_accel, filt] = seeker_ckf(z_meas, filt, dt, params)
% SEEKER_CKF  Cubature Kalman Filter with Singer acceleration model.
%
%   Implements the 3rd-degree spherical-radial cubature rule of
%   Arasaratnam & Haykin (2009) applied to a 6-state LOS estimation
%   problem with Singer's exponentially correlated manoeuvre model.
%
%   State: x = [el; el_dot; el_ddot; az; az_dot; az_ddot]
%          (elevation angle, rate, accel; azimuth angle, rate, accel)
%
%   Singer process model (per axis):
%     dx/dt = [ 0  1     0   ] x + w     w ~ N(0, Q_singer)
%             [ 0  0     1   ]
%             [ 0  0  -1/τ_m ]
%   where τ_m = target manoeuvre time constant,
%         σ_m = RMS manoeuvre acceleration.
%
%   Cubature points (n=6 state → 2n=12 points):
%     ξᵢ = x_hat ± √(n·P) · eᵢ,   weight = 1/(2n)
%
%   References:
%     Arasaratnam & Haykin (2009), "Cubature Kalman Filters",
%       IEEE Trans. Autom. Control 54(6), pp. 1254-1269
%     Singer (1970), "Estimating Optimal Tracking Filter Performance
%       for Manned Maneuvering Targets", IEEE TAC 15(1)
%
%   params fields:
%     .ckf_sigma_m   - Singer RMS manoeuvre acceleration (rad/s^2)
%     .ckf_tau_m     - Singer manoeuvre time constant (s)
%     .ckf_R         - measurement noise variance (rad^2)

    n  = 6;       % state dimension
    n2 = 2*n;     % number of cubature points

    % ---- initialise --------------------------------------------------
    [az_meas, el_meas] = los_to_angles(z_meas);

    if isempty(filt.x)
        filt.x = [el_meas; 0; 0; az_meas; 0; 0];
        filt.P = diag([deg2rad(2)^2, (deg2rad(5))^2, (deg2rad(10))^2, ...
                       deg2rad(2)^2, (deg2rad(5))^2, (deg2rad(10))^2]);
    end

    % ---- Singer state transition matrix (discrete) ------------------
    tau   = params.ckf_tau_m;
    sig_m = params.ckf_sigma_m;
    R_n   = params.ckf_R;

    % Per-axis F (3x3):
    e_tau = exp(-dt/tau);
    F1 = [1,  dt,  tau*(e_tau - 1 + dt/tau);
          0,   1,  1 - e_tau;
          0,   0,  e_tau];

    % Full 6x6 F (block diagonal for el and az axes)
    F = blkdiag(F1, F1);

    % Singer process noise Q per axis:
    q33 = max(0, sig_m^2 * tau*(1 - e_tau^2));          % accel variance
    q22 = max(0, sig_m^2 * (dt - 2*tau*(1-e_tau) + tau*(1 - e_tau^2)/2)); % rate variance
    q11 = max(0, sig_m^2 * dt^5 / 20);                  % angle variance (tiny)
    q23 = sig_m^2 * tau*(1 - e_tau)^2;
    q13 = sig_m^2 * (1 - e_tau)^2 / 2;
    q12 = sig_m^2 * ((1 + e_tau^2)/2 - e_tau) / (tau + eps);
    % Correct ordering: state is [angle; rate; accel] so Q[1,1]=angle, Q[3,3]=accel
    Q1  = [q11, q12, q13;
           q12, q22, q23;
           q13, q23, q33];
    Q   = blkdiag(Q1, Q1) + 1e-10*eye(n);  % nugget for PD

    % Measurement matrix (angles only: el, az)
    H = [1 0 0 0 0 0;
         0 0 0 1 0 0];
    R_mat = R_n * eye(2);

    % =====================================================
    % CUBATURE KALMAN PREDICT
    % =====================================================
    % Cubature points
    try
        S = chol(n * filt.P, 'lower');
    catch
        filt.P = nearestSPD(filt.P);
        S = chol(n * filt.P, 'lower');
    end

    Xi = zeros(n, n2);
    for i = 1:n
        Xi(:,i)   = filt.x + S(:,i);
        Xi(:,i+n) = filt.x - S(:,i);
    end

    % Propagate through dynamics (linear → just F*Xi)
    Xi_pred = F * Xi;

    % Predicted mean and covariance
    x_pred = sum(Xi_pred, 2) / n2;
    P_pred = Q;
    for i = 1:n2
        d = Xi_pred(:,i) - x_pred;
        P_pred = P_pred + d*d' / n2;
    end

    % =====================================================
    % CUBATURE KALMAN UPDATE
    % =====================================================
    try
        S2 = chol(n * P_pred, 'lower');
    catch
        P_pred = nearestSPD(P_pred);
        S2 = chol(n * P_pred, 'lower');
    end

    Xi2 = zeros(n, n2);
    for i = 1:n
        Xi2(:,i)   = x_pred + S2(:,i);
        Xi2(:,i+n) = x_pred - S2(:,i);
    end

    % Measurement prediction (linear → H*Xi2)
    Z_pred = H * Xi2;   % 2×n2
    z_pred = sum(Z_pred, 2) / n2;

    % Innovation covariance
    Pzz = R_mat;
    Pxz = zeros(n, 2);
    for i = 1:n2
        dz = Z_pred(:,i) - z_pred;
        dx = Xi2(:,i)    - x_pred;
        Pzz = Pzz + dz*dz' / n2;
        Pxz = Pxz + dx*dz' / n2;
    end

    % Kalman gain
    K = Pxz / Pzz;

    % Measurement (wrap azimuth innovation)
    z_k      = [el_meas; az_meas];
    innov    = z_k - z_pred;
    innov(2) = angle_wrap(innov(2));

    % Update
    filt.x = x_pred + K * innov;
    filt.P = P_pred - K * Pzz * K';
    filt.P = 0.5*(filt.P + filt.P');   % symmetrise

    % ---- outputs ----------------------------------------------------
    los_vec   = angles_to_los(filt.x(4), filt.x(1));
    los_rate  = los_rate_from_angles(filt.x(4), filt.x(1), ...
                                      filt.x(5), filt.x(2));
    los_accel = los_rate_from_angles(filt.x(4), filt.x(1), ...
                                      filt.x(6), filt.x(3));
end

% ---- helper: nearest symmetric positive-definite matrix -------------
function A = nearestSPD(A)
    A = (A + A')/2;
    [~,p] = chol(A);
    if p > 0
        [V,D] = eig(A);
        d = diag(D);
        d(d < 0) = 1e-10;
        A = V * diag(d) * V';
        A = (A + A')/2;
    end
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
