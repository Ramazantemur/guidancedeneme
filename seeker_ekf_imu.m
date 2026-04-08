function [los_vec, los_rate, filt] = seeker_ekf_imu(los_noisy, imu_rate, filt, dt, params)
% SEEKER_EKF_IMU  Extended Kalman Filter with IMU-aiding — Algorithm 3
%
%   Same structure as SEEKER_EKF_LOS (Algorithm 2) but the PREDICTION step
%   uses the accurate LOS rate measured by the IMU instead of relying purely
%   on the kinematic model.  The measurement remains the noisy uLOS.
%
%   States: x = [el; el_dot; az; az_dot; el_bias; az_bias; imu_bias_el; imu_bias_az]
%           (8-state)  — optional IMU bias augmentation
%
%   IMU provides angular-rate in the body frame.  Here we model it simply
%   as a direct, near-accurate measurement of (el_dot, az_dot) with small
%   additive bias + white noise:
%
%       omega_imu = [el_dot; az_dot] + imu_bias + w_imu
%
%   Prediction:
%     el_{k+1}      = el_k  + dt * omega_imu_el     (use IMU rate)
%     el_dot_{k+1}  = omega_imu_el                  (best estimate of rate)
%     az_{k+1}      = az_k  + dt * omega_imu_az
%     az_dot_{k+1}  = omega_imu_az
%     el_bias_{k+1} = el_bias_k                     (seeker bias RW)
%     az_bias_{k+1} = az_bias_k
%     imu_bias_el_{k+1} = imu_bias_el_k             (IMU bias RW)
%     imu_bias_az_{k+1} = imu_bias_az_k
%
%   Measurement model (same as Alg 2):
%     z_k = [el_k + el_bias_k; az_k + az_bias_k] + v_k
%
%   params fields (in addition to Alg 2 fields):
%     .ekfimu_R0          - measurement noise var (rad^2)
%     .ekfimu_q_rate      - process noise for angle-rate (rad^2/s)
%     .ekfimu_q_bias      - seeker bias noise PSD
%     .ekfimu_q_imu_bias  - IMU bias noise PSD
%     .ekfimu_imu_noise   - IMU white-noise sigma (rad/s)  [for R_imu, not used directly]
%     .ekfimu_p0_ang      - init angle sigma  (rad)
%     .ekfimu_p0_rate     - init rate sigma   (rad/s)
%     .ekfimu_p0_bias     - init seeker bias sigma (rad)
%     .ekfimu_p0_imu_bias - init IMU bias sigma (rad/s)
%
%   Inputs:
%     los_noisy : (3x1) noisy unit-LOS vector from laser
%     imu_rate  : (2x1) [el_dot_imu; az_dot_imu]  IMU angular rate (rad/s)
%                  — assumed available from IMU; near-accurate with small bias

    n_x = 8;
    n_z = 2;

    % ---- convert measurement to angles ----------------------------------------
    [az_m, el_m] = los_to_angles(los_noisy);
    z_k = [el_m; az_m];

    % ---- IMU rate (projected to seeker angles) ---------------------------------
    omega_el = imu_rate(1);   % elevation-rate from IMU
    omega_az = imu_rate(2);   % azimuth-rate  from IMU

    % ---- initialise -----------------------------------------------------------
    if isempty(filt.x)
        filt.x = [el_m; omega_el; az_m; omega_az; 0; 0; 0; 0];
        filt.P = diag([params.ekfimu_p0_ang,  params.ekfimu_p0_rate, ...
                       params.ekfimu_p0_ang,  params.ekfimu_p0_rate, ...
                       params.ekfimu_p0_bias, params.ekfimu_p0_bias, ...
                       params.ekfimu_p0_imu_bias, params.ekfimu_p0_imu_bias].^2);
        filt.R = params.ekfimu_R0 * eye(n_z);
    end

    % ---- obtain IMU bias estimate from state for corrected rate ---------------
    ib_el = filt.x(7);
    ib_az = filt.x(8);
    omega_el_c = omega_el - ib_el;   % bias-corrected IMU elevation rate
    omega_az_c = omega_az - ib_az;   % bias-corrected IMU azimuth rate

    % ---- state transition (IMU-driven) ----------------------------------------
    % Use IMU to propagate angle; el_dot set to corrected IMU rate directly
    F = eye(n_x);
    % angle updated by IMU rate × dt
    F(1,1) = 1;  F(1,2) = 0;   % el += dt * omega_el_c  (handled via u)
    F(3,3) = 1;  F(3,4) = 0;   % az += dt * omega_az_c

    % IMU bias and seeker bias are random-walk; el_dot/az_dot set from IMU
    x_p = filt.x;
    x_p(1) = filt.x(1) + dt * omega_el_c;   % integrate IMU rate
    x_p(2) = omega_el_c;                      % best rate estimate
    x_p(3) = filt.x(3) + dt * omega_az_c;
    x_p(4) = omega_az_c;
    % biases: random walk (unchanged)

    % ---- process noise --------------------------------------------------------
    % When the IMU drives the prediction, angle propagation error comes mainly
    % from IMU noise (not from a kinematic model mismatch), so the angle
    % process noise should reflect accumulated IMU noise over dt.
    % q_rate models remaining model error (not IMU noise — that is captured
    % by the IMU noise sigma passed as measurement).
    imu_sig  = params.ekfimu_imu_noise;   % IMU rate noise sigma (rad/s)
    q_b  = params.ekfimu_q_bias;
    q_ib = params.ekfimu_q_imu_bias;
    q_r  = params.ekfimu_q_rate;          % residual rate model error
    % Angle uncertainty from IMU propagation: sigma_ang = imu_sig * sqrt(dt)
    q_ang = imu_sig^2 * dt;              % variance of el/az due to IMU noise
    Q = diag([q_ang, q_r+imu_sig^2, q_ang, q_r+imu_sig^2, ...
              q_b*dt, q_b*dt, q_ib*dt, q_ib*dt]);

    % ---- Jacobian for covariance propagation (F_jac) --------------------------
    F_jac = eye(n_x);
    % angle depends on IMU bias (through the bias correction in x_p computation)
    F_jac(1,7) = -dt;   % del x(1)/del imu_bias_el
    F_jac(2,7) = -1;    % del x(2)/del imu_bias_el
    F_jac(3,8) = -dt;
    F_jac(4,8) = -1;

    P_p = F_jac * filt.P * F_jac' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- measurement matrix ---------------------------------------------------
    H = [1, 0, 0, 0, 1, 0, 0, 0;   % el + el_bias
         0, 0, 1, 0, 0, 1, 0, 0];  % az + az_bias

    % ---- UPDATE ---------------------------------------------------------------
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

    % ---- outputs --------------------------------------------------------------
    el_est     = filt.x(1);
    el_dot_est = filt.x(2);
    az_est     = filt.x(3);
    az_dot_est = filt.x(4);

    los_vec  = angles_to_los(az_est, el_est);
    los_rate = los_rate_from_angles(az_est, el_est, az_dot_est, el_dot_est);

    % Safety: clamp |los_rate| to avoid guidance blow-up on initialisation
    los_rate_mag = norm(los_rate);
    max_los_rate = deg2rad(30);   % 30 deg/s hard limit
    if los_rate_mag > max_los_rate
        los_rate = los_rate * (max_los_rate / los_rate_mag);
    end
end   % seeker_ekf_imu

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
