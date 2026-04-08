function [los_vec, los_rate, filt] = seeker_ekf_3d_imu(los_noisy, imu_rate_3d, filt, dt, params)
% SEEKER_EKF_3D_IMU — Algorithm 10: Dual-Measurement Linear KF (uLOS + IMU rate)
%
%   Same 9-state structure as Algorithm 9:
%     x = [l(3);  l_dot(3);  b(3)]
%
%   KEY DIFFERENCE: applies TWO measurements at each step (sequentially):
%
%   ┌── Measurement 1: Laser (uLOS) ────────────────────────────────────────
%   │   z1 = los_noisy ≈ l + b + v1     (accurate angle, has bias, high noise)
%   │   H1 = [I3, O3, I3]
%   │   R1 = ekf3dimu_R0 * I3           (laser noise, per axis)
%   └────────────────────────────────────────────────────────────────────────
%   ┌── Measurement 2: IMU (uLOS rate) ─────────────────────────────────────
%   │   z2 = imu_rate_3d ≈ l_dot + w2  (near-accurate rate, small noise)
%   │   H2 = [O3, I3, O3]
%   │   R2 = ekf3dimu_R_imu * I3       (IMU noise, per axis, << R1)
%   └────────────────────────────────────────────────────────────────────────
%
%   Sequential update (Joseph form for numerical stability):
%     After prediction, apply laser update → intermediate state x_mid, P_mid
%     Then apply IMU update → final state x_up, P_up
%
%   Why sequential dual-measurement is powerful here:
%   ┌──────────────────────────────────────────────────────────────────────┐
%   │  Laser corrects l and b    (angle + bias estimation)                │
%   │  IMU corrects l_dot        (rate estimation, very directly)         │
%   │  → Full state observability in every step                           │
%   │  → No model-only rate propagation → faster convergence              │
%   │  → Biases separate cleanly from rates                               │
%   └──────────────────────────────────────────────────────────────────────┘
%
%   The IMU provides the 3D dl/dt computed from kinematics:
%     l_dot_true = (v_rel - (v_rel · l̂) · l̂) / r   where v_rel = -v_uav
%   This is the true rate + small noise (sigma = ekf3dimu_imu_noise_3d, rad/s per axis).
%
%   Inputs:
%     los_noisy    : (3×1) noisy unit-LOS vector from laser
%     imu_rate_3d  : (3×1) 3D uLOS rate from IMU [dl/dt, rad/s]
%
%   params fields:
%     .ekf3dimu_R0           - laser meas noise variance per axis (rad^2)
%     .ekf3dimu_R_imu        - IMU rate noise variance per axis (rad/s)^2
%     .ekf3dimu_q_rate       - l_dot process noise PSD (rad^2/s per axis)
%     .ekf3dimu_q_bias       - bias process noise PSD (rad^2/s per axis)
%     .ekf3dimu_p0_pos       - initial l uncertainty sigma
%     .ekf3dimu_p0_rate      - initial l_dot sigma (rad/s)
%     .ekf3dimu_p0_bias      - initial bias sigma (rad)
%     .ekf3dimu_imu_noise_3d - IMU 3D rate noise sigma (rad/s per axis)

    n_x = 9;

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        filt.x  = [los_noisy; zeros(3,1); zeros(3,1)];
        filt.P  = blkdiag(params.ekf3dimu_p0_pos^2  * eye(3), ...
                          params.ekf3dimu_p0_rate^2 * eye(3), ...
                          params.ekf3dimu_p0_bias^2 * eye(3));
        filt.R1 = params.ekf3dimu_R0  * eye(3);   % laser noise
        filt.R2 = params.ekf3dimu_R_imu * eye(3); % IMU rate noise
    end

    % ---- State transition & process noise -----------------------------------
    F = [eye(3),   dt*eye(3), zeros(3);
         zeros(3), eye(3),    zeros(3);
         zeros(3), zeros(3),  eye(3)];

    q_r = params.ekf3dimu_q_rate;
    q_b = params.ekf3dimu_q_bias;
    Q   = blkdiag(q_r*(dt^3/3)*eye(3), q_r*dt*eye(3), q_b*dt*eye(3));

    % ---- PREDICT ------------------------------------------------------------
    x_p = F * filt.x;
    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ====================================================================
    % MEASUREMENT 1: Laser uLOS measurement
    %   z1 = l + b + noise,   H1 = [I3, O3, I3]
    % ====================================================================
    H1 = [eye(3), zeros(3), eye(3)];
    z1_pred = H1 * x_p;
    innov1  = los_noisy - z1_pred;

    S1  = H1 * P_p * H1' + filt.R1;
    K1  = P_p * H1' / S1;

    x_mid = x_p + K1 * innov1;
    P_mid = (eye(n_x) - K1*H1) * P_p * (eye(n_x) - K1*H1)' + K1*filt.R1*K1';
    P_mid = 0.5*(P_mid + P_mid');

    % ====================================================================
    % MEASUREMENT 2: IMU uLOS rate measurement
    %   z2 = l_dot + noise,   H2 = [O3, I3, O3]
    % ====================================================================
    H2 = [zeros(3), eye(3), zeros(3)];
    z2_pred = H2 * x_mid;
    innov2  = imu_rate_3d - z2_pred;

    S2  = H2 * P_mid * H2' + filt.R2;
    K2  = P_mid * H2' / S2;

    x_up = x_mid + K2 * innov2;
    P_up = (eye(n_x) - K2*H2) * P_mid * (eye(n_x) - K2*H2)' + K2*filt.R2*K2';
    P_up = 0.5*(P_up + P_up');

    filt.x = x_up;
    filt.P = P_up;

    % ---- Outputs (sphere constraints on outputs only) -----------------------
    l_est  = filt.x(1:3);
    ld_est = filt.x(4:6);

    r = norm(l_est);
    if r < 1e-6, r = 1e-6; end
    los_vec = l_est / r;

    % Project l_dot onto tangent space (dl/dt ⊥ l)
    los_rate = ld_est - dot(ld_est, los_vec) * los_vec;
end
