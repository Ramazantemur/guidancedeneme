function [los_vec, los_rate, filt] = seeker_ekf_los(los_noisy, filt, dt, params)
% SEEKER_EKF_LOS  Extended Kalman Filter — Algorithm 2
%
%   States:  x = [el; el_dot; az; az_dot; el_bias; az_bias]   (6-state)
%   Measurement: z = [el_meas; az_meas]  (noisy uLOS decomposed to angles)
%
%   This EKF estimates uLOS and its rate, as well as persistent SEEKER
%   elevation and azimuth biases.  No IMU information is used.
%
%   Process model (constant-rate + bias random walk):
%     el_{k+1}      = el_k      + dt * el_dot_k
%     el_dot_{k+1}  = el_dot_k                     (nearly-constant rate)
%     az_{k+1}      = az_k      + dt * az_dot_k
%     az_dot_{k+1}  = az_dot_k
%     el_bias_{k+1} = el_bias_k                    (slowly drifting bias)
%     az_bias_{k+1} = az_bias_k
%
%   Measurement model:
%     z_k = [el_k + el_bias_k; az_k + az_bias_k] + v_k
%
%   params fields:
%     .ekflos_R0       - initial meas noise variance  (rad^2)
%     .ekflos_q_rate   - process noise for angle-rate (rad^2/s)
%     .ekflos_q_bias   - process noise for bias drift  (rad^2/s)
%     .ekflos_p0_ang   - initial angle error  (rad)
%     .ekflos_p0_rate  - initial rate error   (rad/s)
%     .ekflos_p0_bias  - initial bias error   (rad)

    n_x = 6;
    n_z = 2;

    % ---- convert noisy LOS unit vector → elevation / azimuth angles ----------
    [az_m, el_m] = los_to_angles(los_noisy);
    z_k = [el_m; az_m];   % measurement vector

    % ---- initialise on first call -------------------------------------------
    if isempty(filt.x)
        filt.x = [el_m; 0; az_m; 0; 0; 0];
        filt.P = diag([params.ekflos_p0_ang,  params.ekflos_p0_rate, ...
                       params.ekflos_p0_ang,  params.ekflos_p0_rate, ...
                       params.ekflos_p0_bias, params.ekflos_p0_bias].^2);
        filt.R = params.ekflos_R0 * eye(n_z);
    end

    % ---- state transition matrix F ------------------------------------------
    F = [1, dt,  0,  0,  0,  0;   % el
         0,  1,  0,  0,  0,  0;   % el_dot
         0,  0,  1, dt,  0,  0;   % az
         0,  0,  0,  1,  0,  0;   % az_dot
         0,  0,  0,  0,  1,  0;   % el_bias
         0,  0,  0,  0,  0,  1];  % az_bias

    % ---- process noise matrix Q ---------------------------------------------
    q_r = params.ekflos_q_rate;    % rate-noise PSD
    q_b = params.ekflos_q_bias;    % bias-noise PSD
    Q = diag([q_r*dt^3/3, q_r*dt, q_r*dt^3/3, q_r*dt, q_b*dt, q_b*dt]);

    % ---- measurement matrix H -----------------------------------------------
    % z = [el + el_bias; az + az_bias]
    H = [1, 0, 0, 0, 1, 0;
         0, 0, 1, 0, 0, 1];

    % ---- PREDICT ------------------------------------------------------------
    x_p = F * filt.x;
    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- UPDATE -------------------------------------------------------------
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

    % ---- outputs: decode state back to LOS vector & rate -------------------
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
