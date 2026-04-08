function [los_vec, los_rate, target_est, filt] = seeker_ekf_relpos(los_noisy, uav_pos, uav_vel, filt, dt, params)
% SEEKER_EKF_RELPOS  EKF with Relative-Position States — Algorithm 4
%
%   Tracks the RELATIVE position and velocity of the TARGET w.r.t. the UAV.
%   Provides a Cartesian estimate of target position for direct guidance.
%
%   States: x = [drx; dry; drz; dvx; dvy; dvz]   (6-state, NED)
%     dr = r_target - r_uav   (relative position, m)
%     dv = v_target - v_uav   (relative velocity, m/s)
%
%   Process model (target stationary, UAV maneuvering):
%     dr_{k+1} = dr_k + dt*dv_k
%     dv_{k+1} = dv_k - dt*a_uav   (UAV accel changes relative vel)
%
%   Nonlinear measurement model:
%     z_k = los_meas = dr_k / ||dr_k||   (unit-vector, 3 components)
%
%   Jacobian of z w.r.t. dr (outer-product form, rank-2 matrix):
%     H_dr = (I - l*l') / r   where l = dr/r
%
%   RANGE OBSERVABILITY:
%     Bearing-only measurement is inherently weakly observable for range.
%     Three design choices improve convergence significantly:
%       1. Smart range initialisation from altitude geometry
%       2. Altitude pseudo-measurement (target on ground: dr_z = -uav_z)
%       3. Correct los_rate formula: dL/dt = (dv - (dv·l̂)l̂) / r
%
%   params fields:
%     .ekfrp_R0        - meas noise variance per axis (rad^2)
%     .ekfrp_q_pos     - position process noise PSD (m^2/s^3)
%     .ekfrp_q_vel     - velocity process noise PSD (m^2/s^3)
%     .ekfrp_p0_pos    - initial position sigma per axis (m)
%     .ekfrp_p0_vel    - initial velocity sigma per axis (m/s)
%     .ekfrp_use_alt   - (optional, default true) use altitude pseudo-meas

    n_x = 6;
    n_z = 3;   % full 3-D unit-LOS measurement

    % ---- initialise -----------------------------------------------------------
    if isempty(filt.x)
        % --- Smart range init from altitude constraint ---------------------
        % Target is near ground (z ≈ 0 NED). UAV is at z = uav_pos(3) < 0.
        % The vertical component of the LOS vector: los_z = dr_z / r
        % Since target at z=0: dr_z = -uav_pos(3) > 0 (pointing down in NED)
        % Therefore: r_init = dr_z / los_z = -uav_pos(3) / los_noisy(3)
        %            (guard against near-zero los_z)
        if los_noisy(3) > 0.05    % at least ~3° depression angle
            r_init = -uav_pos(3) / los_noisy(3);
            r_init = max(200, min(5000, r_init));  % clamp to plausible range
        else
            r_init = 1500;         % fallback if nearly horizontal LOS
        end

        dr_init = los_noisy * r_init;
        filt.x  = [dr_init; -uav_vel];   % dv ≈ -v_uav (stationary target)
        filt.P  = diag([params.ekfrp_p0_pos, params.ekfrp_p0_pos, params.ekfrp_p0_pos, ...
                        params.ekfrp_p0_vel, params.ekfrp_p0_vel, params.ekfrp_p0_vel].^2);
        filt.R  = params.ekfrp_R0 * eye(n_z);
        filt.vel_prev = uav_vel;
        % Altitude pseudo-meas noise (m^2) — generous, target ~on-ground
        filt.R_alt = 5.0^2;
    end

    % ---- UAV acceleration estimate (finite-difference from velocity) ----------
    if ~isempty(filt.vel_prev)
        a_uav = (uav_vel - filt.vel_prev) / dt;
    else
        a_uav = zeros(3,1);
    end
    filt.vel_prev = uav_vel;

    % ---- PREDICT --------------------------------------------------------------
    F = [eye(3), dt*eye(3);
         zeros(3), eye(3)];

    u_input = [zeros(3,1); -dt * a_uav];   % UAV accel drives relative vel
    x_p = F * filt.x + u_input;

    q_p = params.ekfrp_q_pos;
    q_v = params.ekfrp_q_vel;
    Q = diag([q_p*dt, q_p*dt, q_p*dt, q_v*dt, q_v*dt, q_v*dt]);

    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- LOS measurement update (nonlinear EKF) --------------------------------
    dr = x_p(1:3);
    r  = norm(dr);
    if r < 0.5, r = 0.5; end       % guard division-by-zero near target

    z_pred = dr / r;
    % Jacobian: d(dr/r)/d(dr) = (I - l*l') / r   (3x3, rank-2 — correct!)
    H_dr = (eye(3) - (dr * dr') / r^2) / r;
    H = [H_dr, zeros(3,3)];

    % Innovation: plain difference — DO NOT project out the range component!
    % (That projection was the original bug: it destroyed range observability.)
    innov = los_noisy - z_pred;

    S = H * P_p * H' + filt.R;
    K = P_p * H' / S;

    x_up = x_p + K * innov;
    P_up = (eye(n_x) - K*H) * P_p * (eye(n_x) - K*H)' + K*filt.R*K';
    P_up = 0.5*(P_up + P_up');

    % ---- Altitude pseudo-measurement (target on ground, dr_z = -uav_pos(3)) --
    %   z_alt = [0 0 1 0 0 0] * x = dr_z
    %   z_alt_meas = 0 - uav_pos(3)  = -uav_pos(3)  (= height AGL in NED)
    use_alt = ~isfield(params, 'ekfrp_use_alt') || params.ekfrp_use_alt;
    if use_alt
        H_alt = [0, 0, 1, 0, 0, 0];           % observe dr_z only
        z_alt_pred = H_alt * x_up;             % after LOS update
        z_alt_meas = -uav_pos(3);              % expected dr_z if target at z=0
        innov_alt  = z_alt_meas - z_alt_pred;

        S_alt = H_alt * P_up * H_alt' + filt.R_alt;
        K_alt = P_up * H_alt' / S_alt;

        x_up = x_up + K_alt * innov_alt;
        P_up = (eye(n_x) - K_alt * H_alt) * P_up;
        P_up = 0.5*(P_up + P_up');
    end

    filt.x = x_up;
    filt.P = P_up;

    % ---- Outputs ---------------------------------------------------------------
    dr_est = filt.x(1:3);
    dv_est = filt.x(4:6);

    r_est = norm(dr_est);
    if r_est < 0.1, r_est = 0.1; end

    los_vec = dr_est / r_est;   % estimated unit-LOS direction

    % FIXED los_rate formula: dl/dt = (dv - (dv·l̂)·l̂) / r
    % This is the time-derivative of the unit-LOS vector (consistent with
    % the rest of the codebase: seeker_vbaekf, bpng_guidance, etc.)
    % The old cross(dr,dv)/r^2 was the angular velocity VECTOR ω, not dl/dt.
    l_hat     = los_vec;
    dv_perp   = dv_est - dot(dv_est, l_hat) * l_hat;   % component ⊥ LOS
    los_rate  = dv_perp / r_est;

    % Estimated target position in NED absolute coordinates
    target_est = uav_pos + dr_est;
end
