function [los_vec, los_rate, filt] = seeker_ekf_3d(los_noisy, filt, dt, params)
% SEEKER_EKF_3D — Algorithm 9: Linear KF in 3D Cartesian uLOS space
%
%   States: x = [l(3);  l_dot(3);  b(3)]   (9-state)
%     l     : uLOS unit vector (3D Cartesian, NED)    — unconstrained in filter
%     l_dot : d(uLOS)/dt = LOS rate (3D, rad/s on tangent sphere)
%     b     : persistent measurement bias on l (3D seeker bias)
%
%   *** This filter is entirely LINEAR — no EKF linearisation ***
%   (Both the process model and measurement model are linear in the states.)
%
% ─────────────────── Process model ─────────────────────────────────────────
%   l_{k+1}     = l_k  + dt * l_dot_k     (integrate rate)
%   l_dot_{k+1} = l_dot_k                 (nearly constant LOS rate)
%   b_{k+1}     = b_k                     (random walk)
%
%   F = [I3, dt*I3, O3;
%         O3,   I3, O3;
%         O3,   O3, I3]
%
% ─────────────────── Measurement model ──────────────────────────────────────
%   z = los_noisy ≈ l + b + v_k
%   H = [I3, O3, I3]    (3×9 matrix)
%
% ─────────────────── Why this formulation? ───────────────────────────────────
%   Working in 3D Cartesian space instead of (el, az) angles:
%   + No gimbal-lock / singularity near vertical or horizontal LOS
%   + Bias is naturally 3D (can capture any direction offset)
%   + Fully linear → standard KF (no Jacobian approximation)
%   + Rate state is directly the physical 3D derivative of uLOS
%   - The unit-sphere constraint ||l||=1 is NOT enforced during filtering
%     (output is renormalised, and tangent-space projection applied to rate)
%
% ─────────────────── Outputs ─────────────────────────────────────────────────
%   los_vec  : l_est / ||l_est||     (renormalised unit LOS vector)
%   los_rate : l_dot_est projected onto tangent space of sphere at los_vec
%              so that dl/dt is always perpendicular to l (physically correct)
%
% params fields:
%   .ekf3d_R0        - laser measurement noise variance per axis (rad^2)
%   .ekf3d_q_rate    - l_dot process noise PSD (rad^2/s per axis)
%   .ekf3d_q_bias    - bias process noise PSD (rad^2/s per axis)
%   .ekf3d_p0_pos    - initial l  uncertainty sigma (unitless)
%   .ekf3d_p0_rate   - initial l_dot uncertainty sigma (rad/s)
%   .ekf3d_p0_bias   - initial bias uncertainty sigma (rad)

    n_x = 9;
    n_z = 3;

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        filt.x = [los_noisy; zeros(3,1); zeros(3,1)];
        filt.P = blkdiag(params.ekf3d_p0_pos^2  * eye(3), ...
                         params.ekf3d_p0_rate^2 * eye(3), ...
                         params.ekf3d_p0_bias^2 * eye(3));
        filt.R = params.ekf3d_R0 * eye(n_z);
    end

    % ---- State transition & process noise -----------------------------------
    F = [eye(3),   dt*eye(3), zeros(3);
         zeros(3), eye(3),    zeros(3);
         zeros(3), zeros(3),  eye(3)];

    q_r = params.ekf3d_q_rate;
    q_b = params.ekf3d_q_bias;
    Q = blkdiag(q_r*(dt^3/3)*eye(3), ...   % pos: small (driven by integrated rate)
                q_r*dt       *eye(3), ...   % rate: main process noise
                q_b*dt       *eye(3));      % bias: slow random walk

    % ---- Measurement matrix (linear) ----------------------------------------
    H = [eye(3), zeros(3), eye(3)];   % z = l + b

    % ---- PREDICT (standard KF) -----------------------------------------------
    x_p = F * filt.x;
    P_p = F * filt.P * F' + Q;
    P_p = 0.5*(P_p + P_p');

    % ---- UPDATE -------------------------------------------------------------
    z_pred = H * x_p;
    innov  = los_noisy - z_pred;

    S = H * P_p * H' + filt.R;
    K = P_p * H' / S;

    x_up = x_p + K * innov;
    P_up = (eye(n_x) - K*H) * P_p * (eye(n_x) - K*H)' + K*filt.R*K';
    P_up = 0.5*(P_up + P_up');

    filt.x = x_up;
    filt.P = P_up;

    % ---- Outputs (with sphere constraints applied to outputs only) -----------
    l_est    = filt.x(1:3);
    ld_est   = filt.x(4:6);

    % Renormalise l to unit sphere
    r = norm(l_est);
    if r < 1e-6, r = 1e-6; end
    los_vec = l_est / r;

    % Project l_dot onto tangent space: dl/dt must be perpendicular to l
    % (The KF may have a small radial component due to linear approximation;
    %  projecting removes the physically impossible radial part.)
    los_rate = ld_est - dot(ld_est, los_vec) * los_vec;
end
