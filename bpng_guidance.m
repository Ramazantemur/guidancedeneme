function accel_cmd = bpng_guidance(state, target_pos, los_rate, params)
% BPNG_GUIDANCE  Biased Proportional Navigation Guidance law.
%
%   Combines Pure Pursuit (PP) and PNG, blended by LOS alignment:
%     - When V is MISALIGNED with LOS (large angle-off), Pure Pursuit steers
%       the velocity toward the LOS direction (ensures guidance always works).
%     - When V is ALIGNED with LOS (small angle-off), standard PNG dominates
%       for terminal accuracy.
%
%   Both terms are projected LATERAL (perpendicular to V) so that the
%   constant-speed hold does not cancel the guidance command.
%
%   Acceleration command:
%     a_cmd = (1 - w) * PP_term + w * PNG_term + bias
%     where w = max(0, V_hat · LOS_hat)  (0 misaligned, 1 aligned)
%
%   Inputs
%     state      : [x;y;z;Vx;Vy;Vz]   UAV state (NED)
%     target_pos : [xt;yt;zt]          target position (NED, m)
%     los_rate   : [wx;wy;wz]          LOS angular rate (rad/s) from seeker
%     params     : struct with fields
%                    N      - navigation constant (3-5)
%                    V_ref  - airspeed (m/s)
%                    bias   - [bx;by;bz] bias acceleration (m/s^2), NED
%                    a_max  - saturation limit (m/s^2)
%
%   Output
%     accel_cmd  : [ax;ay;az]  commanded lateral acceleration (m/s^2, NED)

    uav_pos = state(1:3);
    uav_vel = state(4:6);
    V_mag = norm(uav_vel);
    if V_mag < 1e-3, V_mag = 1e-3; end
    V_hat = uav_vel / V_mag;

    % ---- geometry -------------------------------------------------------
    r_vec = target_pos - uav_pos;
    range = norm(r_vec);
    if range < 0.1, range = 0.1; end
    los_hat = r_vec / range;           % unit LOS vector

    % ---- alignment weight -----------------------------------------------
    % w = 0: V perpendicular to LOS (use Pure Pursuit)
    % w = 1: V aligned with LOS     (use PNG)
    w = max(0, dot(V_hat, los_hat));

    % ---- PNG term: N * V * omega_LOS_lateral ----------------------------
    % Project LOS rate perpendicular to velocity (lateral only)
    los_rate_lat = los_rate - dot(los_rate, V_hat) * V_hat;
    png_term = params.N * V_mag * los_rate_lat;

    % ---- Pure pursuit term: steer V toward LOS --------------------------
    % Component of LOS hat perpendicular to V — this is the "rotation needed"
    los_hat_lat = los_hat - dot(los_hat, V_hat) * V_hat;
    pursuit_term = params.N * V_mag * los_hat_lat;

    % ---- Blend + bias ---------------------------------------------------
    accel_cmd = (1 - w) * pursuit_term + w * png_term + params.bias;

    % ---- Saturation -----------------------------------------------------
    a_mag = norm(accel_cmd);
    if a_mag > params.a_max
        accel_cmd = accel_cmd * (params.a_max / a_mag);
    end
end
