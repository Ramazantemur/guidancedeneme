function state_dot = uav_dynamics(~, state, accel_cmd, params)
% UAV_DYNAMICS  Point-mass 3-DOF equations of motion (NED frame)
%
%   State vector:  [x; y; z; Vx; Vy; Vz]  (metres, metres/sec)
%   accel_cmd   :  [ax; ay; az] commanded acceleration (m/s^2), NED
%   params      :  struct with fields: V_ref (m/s), g (m/s^2)
%
%   The speed controller is assumed ideal: after integrating, the velocity
%   vector is normalised back to V_ref so airspeed stays constant.
%
%   Returns state_dot = [Vx; Vy; Vz; ax; ay; az]

    % ---- unpack --------------------------------------------------------
    Vx = state(4);
    Vy = state(5);
    Vz = state(6);

    ax = accel_cmd(1);
    ay = accel_cmd(2);
    az = accel_cmd(3);

    % ---- equations of motion -------------------------------------------
    state_dot = zeros(6,1);
    state_dot(1) = Vx;      % dx/dt
    state_dot(2) = Vy;      % dy/dt
    state_dot(3) = Vz;      % dz/dt
    state_dot(4) = ax;      % dVx/dt
    state_dot(5) = ay;      % dVy/dt
    state_dot(6) = az;      % dVz/dt
end
