function [los_vec, los_rate, filt] = seeker_ekf_imu_prop(los_noisy, imu_rate, filt, dt, params)
% SEEKER_EKF_IMU_PROP — Algorithm 5: EKF-LOS with direct IMU-rate propagation
%
%   Same state space as Algorithm 2 (EKF-LOS):
%     x = [el; el_dot; az; az_dot; el_bias; az_bias]   (6-state)
%
%   KEY DIFFERENCE vs Alg 2: the rate states (el_dot, az_dot) are set
%   DIRECTLY from the IMU measurement in the prediction step, NOT from
%   the kinematic model  el_dot_k+1 = el_dot_k.
%
%   KEY DIFFERENCE vs Alg 3 (EKF-IMU): Alg 3 adds IMU bias states (8-state)
%   and estimates them online.  Alg 5 keeps only 6 states — the IMU is
%   treated as a near-perfect control input.
%
%   Prediction (IMU drives rate):
%     el_{k+1}      = el_k  + dt * omega_imu_el      (integrate IMU)
%     el_dot_{k+1}  = omega_imu_el                   (set from IMU)
%     az_{k+1}      = az_k  + dt * omega_imu_az
%     az_dot_{k+1}  = omega_imu_az
%     el_bias, az_bias = random walk (unchanged)
%
%   F Jacobian (deri of prediction w.r.t. state):
%     Angle states depend only on themselves (NOT on old rate state).
%     Rate rows are zero (rates come entirely from IMU, not prior state).
%     F = [1 0 0 0 0 0;  0 0 0 0 0 0;  0 0 1 0 0 0;
%          0 0 0 0 0 0;  0 0 0 0 1 0;  0 0 0 0 0 1]
%
%   Measurement: z = [el + el_bias; az + az_bias]  from noisy laser
%
%   Advantages:
%     + Rate states tightly coupled to IMU (fast, accurate)
%     + Seeker bias still estimated from LOS measurement
%     + Simpler than Alg 3 (no IMU-bias states to tune)
%   Trade-offs:
%     - IMU noise enters the rate estimate directly (no smoothing)
%     - No IMU bias correction (assumes IMU is well-calibrated)
%
%   params:
%     .ekfprop_R0        - LOS measurement noise var (rad^2)
%     .ekfprop_imu_noise - IMU rate noise sigma (rad/s)
%     .ekfprop_q_bias    - seeker bias PSD (rad^2/s)
%     .ekfprop_p0_ang    - initial angle sigma (rad)
%     .ekfprop_p0_rate   - initial rate sigma (rad/s)
%     .ekfprop_p0_bias   - initial bias sigma (rad)

    n_x = 6;
    n_z = 2;

    % ---- Measurement: noisy LOS → angles ------------------------------------
    [az_m, el_m] = los_to_angles(los_noisy);
    z_k = [el_m; az_m];

    omega_el = imu_rate(1);   % elevation-rate from IMU (rad/s)
    omega_az = imu_rate(2);   % azimuth-rate   from IMU (rad/s)

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        filt.x = [el_m; omega_el; az_m; omega_az; 0; 0];
        filt.P = diag([params.ekfprop_p0_ang,  params.ekfprop_p0_rate, ...
                       params.ekfprop_p0_ang,  params.ekfprop_p0_rate, ...
                       params.ekfprop_p0_bias, params.ekfprop_p0_bias].^2);
        filt.R = params.ekfprop_R0 * eye(n_z);
    end

    % ---- PREDICT (IMU as control input) --------------------------------------
    x_p    = filt.x;                             % start from previous
    x_p(1) = filt.x(1) + dt * omega_el;          % integrate IMU rate
    x_p(2) = omega_el;                            % rate ← IMU direct
    x_p(3) = filt.x(3) + dt * omega_az;
    x_p(4) = omega_az;
    % x_p(5:6) = biases, unchanged (random walk)

    % Jacobian of x_p w.r.t. x_prev
    %   El/az: depend on their own previous value (NOT on old rate state,
    %          because dt*omega is an EXTERNAL input, not dt*x(2))
    %   Rate rows: zero — entirely from IMU, no state dependency
    %   Bias rows: identity (random walk)
    F_jac = zeros(n_x, n_x);
    F_jac(1,1) = 1;   % el depends on el_prev
    F_jac(3,3) = 1;   % az depends on az_prev
    F_jac(5,5) = 1;   % el_bias random walk
    F_jac(6,6) = 1;   % az_bias random walk
    % Rows 2 and 4 (el_dot, az_dot) are zero — IMU breaks state dependency

    % Process noise: angle uncertainty from IMU noise integration
    imu_sig = params.ekfprop_imu_noise;
    q_b     = params.ekfprop_q_bias;
    Q = diag([imu_sig^2*dt, imu_sig^2, imu_sig^2*dt, imu_sig^2, ...
              q_b*dt, q_b*dt]);

    P_p = F_jac * filt.P * F_jac' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- MEASUREMENT UPDATE --------------------------------------------------
    H = [1, 0, 0, 0, 1, 0;   % el_meas = el + el_bias
         0, 0, 1, 0, 0, 1];  % az_meas = az + az_bias

    z_pred = H * x_p;
    innov  = z_k - z_pred;
    innov(2) = angle_wrap(innov(2));

    S = H * P_p * H' + filt.R;
    K = P_p * H' / S;

    x_up = x_p + K * innov;
    P_up = (eye(n_x) - K*H) * P_p * (eye(n_x) - K*H)' + K*filt.R*K';
    P_up = 0.5*(P_up + P_up');

    filt.x = x_up;
    filt.P = P_up;

    % ---- Outputs ------------------------------------------------------------
    el_est     = filt.x(1);
    el_dot_est = filt.x(2);   % IMU-propagated, LOS-corrected
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
