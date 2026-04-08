function accel_cmd = circular_autopilot(state, params)
% CIRCULAR_AUTOPILOT  Lateral + vertical acceleration commands to hold a
%                     horizontal circular orbit.
%
%   The autopilot computes:
%     - Centripetal lateral acceleration to track the orbit circle
%     - A proportional altitude-hold term to keep constant z
%
%   Inputs
%     state  : [x; y; z; Vx; Vy; Vz]  current UAV state (NED)
%     params : struct with fields
%                cx, cy  - orbit centre (m, NED)
%                R       - orbit radius (m)
%                h_ref   - reference altitude (m, positive up → z = -h)
%                V_ref   - reference airspeed (m/s)
%                g       - gravitational acceleration (m/s^2)
%                k_alt   - altitude hold gain (rad/s^2 per metre)
%                k_lat   - lateral path-following gain
%
%   Output
%     accel_cmd : [ax; ay; az]  commanded acceleration, NED (m/s^2)

    x  = state(1);  y  = state(2);  z  = state(3);
    Vx = state(4);  Vy = state(5);  Vz = state(6);

    % ---- vector from centre to aircraft (horizontal) -------------------
    dx = x - params.cx;
    dy = y - params.cy;
    r  = sqrt(dx^2 + dy^2);          % current radius

    % ---- desired centripetal acceleration (inward) ---------------------
    % unit vector pointing from aircraft toward centre
    if r < 1e-3
        r = 1e-3;
    end
    e_in = [-dx/r; -dy/r];           % inward unit vector (NED x-y)

    % radius error: positive → too far out
    r_err = r - params.R;

    % lateral accel = centripetal + proportional radius correction
    a_centripetal = params.V_ref^2 / params.R;
    a_lat_xy = (a_centripetal + params.k_lat * r_err) * e_in;

    % ---- altitude hold (NED: z positive downward, altitude = -z) -------
    z_ref  = -params.h_ref;           % desired NED z
    z_err  = z - z_ref;              % positive → too low in NED
    az_cmd = -params.k_alt * z_err - 2*sqrt(params.k_alt) * Vz;

    % ---- assemble command ----------------------------------------------
    accel_cmd = [a_lat_xy(1); a_lat_xy(2); az_cmd];
end
