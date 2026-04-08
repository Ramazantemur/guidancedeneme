%% MAIN_SIM.M  — UAV Guidance + Multi-Filter Seeker Comparison
%
%  Runs the full simulation (circular orbit → BPNG terminal guidance)
%  for each seeker filter across 3 noise/stress scenarios, then prints
%  a comparison table and generates overlay plots.
%
%  Filters tested:
%    'simple'    - 1st-order low-pass (baseline)
%    'alphabeta' - Alpha-Beta-Gamma tracker
%    'std'       - Super-Twisting Differentiator (HOSM)
%    'ckf'       - Cubature Kalman Filter + Singer
%    'vbaekf'    - Variational Bayes Adaptive EKF
%
%  Scenarios:
%    1. Nominal    : noise 0.1 deg
%    2. High noise : noise 1.5 deg
%    3. Very high  : noise 3.0 deg
% =========================================================================
clear; clc; close all;
rng(42);

% =========================================================================
%% BASE SIMULATION PARAMETERS (shared across all runs)
% =========================================================================
base.g        = 9.81;
base.V_ref    = 150;
base.cx       = 0;   base.cy = 0;
base.R        = 1000;
base.h_ref    = 1500;
base.n_laps   = 2;
base.k_lat    = 0.5;
base.k_alt    = 4.0;
base.target_pos  = [0; 0; 0];
base.fov_half_deg   = 45;
base.tau_filter     = 0.1;       % for 'simple' filter only

% BPNG parameters
base.N      = 4;
base.bias   = [0; 0; 0];
base.a_max  = 10 * base.g;

% Simulation steps
base.dt             = 0.05;
base.t_max_guidance = 60;
base.miss_threshold = 3;

% ---- Filter-specific parameters ----------------------------------------
% Alpha-Beta-Gamma
base.abg_wn = 3.0;             % bandwidth (rad/s)

% Super-Twisting Differentiator
base.std_L     = 0.05;         % Lipschitz const (small = less chattering)
base.std_n_sub = 20;           % more sub-steps for stability

% CKF + Singer
base.ckf_sigma_m = 0.05;       % Singer RMS accel (rad/s^2) — realistic for LOS
base.ckf_tau_m   = 3.0;        % Singer time constant (s)
base.ckf_R       = (deg2rad(0.1))^2;   % updated per scenario

% VB-AEKF
base.vb_tau_m   = 3.0;
base.vb_sigma_m = 0.15;        % increased for faster LOS rate tracking
base.vb_R0      = (deg2rad(0.1))^2;
base.vb_nu_R    = 8;           % faster R adaptation
base.vb_nu_Q    = 5;

% =========================================================================
%% SCENARIOS AND FILTERS
% =========================================================================
scenarios = {
    'Nominal (0.1°)',    0.1;
    'High noise (1.5°)', 1.5;
    'Very high (3.0°)',  3.0;
};

filters = {'simple', 'alphabeta', 'std', 'ckf', 'vbaekf'};
filter_labels = {'Simple LP', 'Alpha-Beta-Gamma', 'Super-Twisting', 'CKF+Singer', 'VB-AEKF'};

n_scen = size(scenarios, 1);
n_filt = numel(filters);

% Results storage
miss_dist = nan(n_filt, n_scen);
los_rmse  = nan(n_filt, n_scen);
max_accel = nan(n_filt, n_scen);
engage_t  = nan(n_filt, n_scen);
success   = false(n_filt, n_scen);

% =========================================================================
%% PRE-COMPUTE CIRCULAR PHASE (same for all runs)
% =========================================================================
fprintf('Pre-computing circular orbit phase...\n');
p = base;
p.filter_type = 'simple';

x0 = p.cx + p.R;   y0 = p.cy;   z0 = -p.h_ref;
state0 = [x0; y0; z0; 0; p.V_ref; 0];

period       = 2*pi*p.R / p.V_ref;
n_steps_circ = ceil(p.n_laps * period / p.dt);

state = state0;
for k = 1:n_steps_circ
    accel = circular_autopilot(state, p);
    state = rk4_step(@uav_dynamics, 0, state, accel, p, p.dt);
    V = state(4:6); state(4:6) = V * (p.V_ref / max(norm(V),1e-3));
end
state_handover = state;
fprintf('  Orbit done. Handover at (%.0f, %.0f, %.0f) m\n\n', ...
    state_handover(1), state_handover(2), state_handover(3));

% =========================================================================
%% MAIN COMPARISON LOOP
% =========================================================================
% Store trajectories for plots (first scenario only)
traj_store = cell(n_filt, 1);

for s = 1:n_scen
    noise_deg = scenarios{s, 2};
    fprintf('=== Scenario %d: %s ===\n', s, scenarios{s,1});

    for f = 1:n_filt
        ftype = filters{f};

        % Build params for this run
        p = base;
        p.filter_type    = ftype;
        p.noise_std_deg  = noise_deg;
        % Update noise-dependent initial guesses
        p.ckf_R  = (deg2rad(noise_deg))^2;
        p.vb_R0  = (deg2rad(noise_deg))^2;

        rng(42 + f + s*100);   % reproducible but different seed per run

        % ---- Run guidance phase -------------------------------------
        state  = state_handover;
        seeker = init_seeker(ftype);

        n_steps  = ceil(p.t_max_guidance / p.dt);
        t_arr    = zeros(n_steps+1,1);
        x_arr    = zeros(n_steps+1,3);
        lr_arr   = zeros(n_steps+1,3);  % log LOS rate
        ac_arr   = zeros(n_steps+1,3);  % log accel

        x_arr(1,:) = state(1:3)';
        range_prev  = norm(state(1:3) - p.target_pos);

        k_end  = n_steps;
        hit    = false;
        for k = 1:n_steps
            [los_rate, los_vec, in_fov, seeker] = ...
                laser_seeker(state, p.target_pos, seeker, p.dt, p);

            if in_fov
                accel = bpng_guidance(state, p.target_pos, los_rate, p);
            else
                accel = zeros(3,1);
            end

            state = rk4_step(@uav_dynamics, 0, state, accel, p, p.dt);
            V = state(4:6); state(4:6) = V * (p.V_ref / max(norm(V),1e-3));

            t_arr(k+1)    = k * p.dt;
            x_arr(k+1,:)  = state(1:3)';
            lr_arr(k+1,:) = los_rate';
            ac_arr(k+1,:) = accel';

            range_now = norm(state(1:3) - p.target_pos);
            if range_now < p.miss_threshold || state(3) >= 0
                k_end = k+1;  hit = true;  break;
            end
        end

        % ---- Metrics -----------------------------------------------
        final_pos = state(1:3);
        md = norm(final_pos - p.target_pos);
        miss_dist(f,s) = md;
        engage_t(f,s)  = t_arr(k_end);
        success(f,s)   = hit && (md < 20);

        % Max acceleration (g)
        ac_trim = ac_arr(1:k_end,:);
        max_accel(f,s) = max(vecnorm(ac_trim,2,2)) / p.g;

        % LOS rate RMSE (use magnitude — true rate not easily available)
        lr_trim = lr_arr(2:k_end,:);
        los_rmse(f,s) = mean(vecnorm(lr_trim,2,2));   % mean |omega_LOS|

        % Store first-scenario trajectories for overlay plot
        if s == 1
            traj_store{f}.t    = t_arr(1:k_end);
            traj_store{f}.xyz  = x_arr(1:k_end,:);
            traj_store{f}.lr   = lr_arr(1:k_end,:);
            traj_store{f}.miss = md;
        end

        status_str = 'MISS';
        if success(f,s), status_str = 'HIT'; end
        fprintf('  [%s] %-14s : miss=%.2f m  t=%.1f s  maxA=%.1fg  [%s]\n', ...
            ftype, filter_labels{f}, md, t_arr(k_end), max_accel(f,s), status_str);
    end
    fprintf('\n');
end

% =========================================================================
%% PRINT COMPARISON TABLE
% =========================================================================
fprintf('\n');
fprintf('%s\n', repmat('=',1,80));
fprintf('  FILTER COMPARISON TABLE — Miss Distance (m)\n');
fprintf('%s\n', repmat('=',1,80));

% Header
hdr = sprintf('%-16s', 'Filter');
for s = 1:n_scen
    hdr = [hdr, sprintf('  %-20s', scenarios{s,1})]; %#ok
end
fprintf('%s\n', hdr);
fprintf('%s\n', repmat('-',1,80));

for f = 1:n_filt
    row = sprintf('%-16s', filter_labels{f});
    for s = 1:n_scen
        if success(f,s)
            cell_str = sprintf('%.2f m ✓', miss_dist(f,s));
        else
            cell_str = sprintf('%.0f m ✗', miss_dist(f,s));
        end
        row = [row, sprintf('  %-20s', cell_str)]; %#ok
    end
    fprintf('%s\n', row);
end
fprintf('%s\n', repmat('=',1,80));

fprintf('\n  Max Lateral Acceleration (g)\n');
fprintf('%s\n', repmat('-',1,80));
for f = 1:n_filt
    row = sprintf('%-16s', filter_labels{f});
    for s = 1:n_scen
        row = [row, sprintf('  %-20s', sprintf('%.1f g', max_accel(f,s)))]; %#ok
    end
    fprintf('%s\n', row);
end

fprintf('\n  Mean LOS Rate Magnitude (rad/s) — lower = smoother estimate\n');
fprintf('%s\n', repmat('-',1,80));
for f = 1:n_filt
    row = sprintf('%-16s', filter_labels{f});
    for s = 1:n_scen
        row = [row, sprintf('  %-20s', sprintf('%.4f', los_rmse(f,s)))]; %#ok
    end
    fprintf('%s\n', row);
end
fprintf('%s\n\n', repmat('=',1,80));

% =========================================================================
%% PLOTS — Scenario 1 overlay
% =========================================================================
colors = lines(n_filt);
ls_map = {'-','--','-.',':','--'};

% ---- 3D Trajectory ------------------------------------------------------
figure('Name','3D Trajectories (Nominal Noise)','Color','w','Position',[100 100 900 600]);
hold on; grid on;
for f = 1:n_filt
    xyz = traj_store{f}.xyz;
    plot3(xyz(:,2), xyz(:,1), -xyz(:,3), ...
          'Color', colors(f,:), 'LineStyle', ls_map{f}, 'LineWidth', 1.8, ...
          'DisplayName', sprintf('%s (%.2fm)', filter_labels{f}, traj_store{f}.miss));
end
tp = base.target_pos;
plot3(tp(2), tp(1), -tp(3), 'kp', 'MarkerSize',14,'MarkerFaceColor','k','DisplayName','Target');
xlabel('East (m)'); ylabel('North (m)'); zlabel('Alt (m)');
title('BPNG Terminal Phase — All Filters (Nominal 0.1° noise)');
legend('Location','northeast','FontSize',8); view(30,30);

% ---- LOS Rate magnitude over time --------------------------------------
figure('Name','LOS Rate Estimates (Nominal Noise)','Color','w','Position',[100 120 900 400]);
hold on; grid on;
for f = 1:n_filt
    lr = traj_store{f}.lr;
    t  = traj_store{f}.t;
    plot(t, rad2deg(vecnorm(lr,2,2)), ...
         'Color',colors(f,:),'LineStyle',ls_map{f},'LineWidth',1.5,...
         'DisplayName',filter_labels{f});
end
xlabel('Time (s)'); ylabel('|ω_{LOS}| (deg/s)');
title('LOS Rate Estimate Magnitude — All Filters');
legend('Location','northeast','FontSize',8);

% ---- Miss distance bar chart -------------------------------------------
figure('Name','Miss Distance Comparison','Color','w','Position',[100 140 900 450]);
bar_data = miss_dist';     % [scenarios × filters]
b = bar(bar_data, 'grouped');
for f = 1:n_filt
    b(f).FaceColor = colors(f,:);
end
set(gca,'XTickLabel', {scenarios{:,1}}, 'FontSize', 10);
ylabel('Miss Distance (m)'); title('Miss Distance — All Filters × All Scenarios');
legend(filter_labels, 'Location','northwest','FontSize',8);
grid on; yline(base.miss_threshold,'r--','Threshold','LabelHorizontalAlignment','right');

% =========================================================================
%% HELPER FUNCTIONS
% =========================================================================
function s = init_seeker(ftype)
    s.los_vec_prev   = [];
    s.los_rate_filt  = [];
    s.range          = 0;
    switch lower(ftype)
        case 'alphabeta'
            s.filt_state.x_el = [];
            s.filt_state.x_az = [];
        case 'std'
            s.filt_state.z_el     = [];
            s.filt_state.z_az     = [];
            s.filt_state.z1_el_sm = 0;
            s.filt_state.z1_az_sm = 0;
        case {'ckf','vbaekf'}
            s.filt_state.x  = [];
            s.filt_state.P  = [];
            s.filt_state.R  = [];
            s.filt_state.Q  = [];
            s.filt_state.k  = 1;
        otherwise     % 'simple'
            % nothing extra
    end
end

function state_new = rk4_step(dyn_func, t, state, accel, prm, h)
    k1 = dyn_func(t,     state,         accel, prm);
    k2 = dyn_func(t+h/2, state + h/2*k1, accel, prm);
    k3 = dyn_func(t+h/2, state + h/2*k2, accel, prm);
    k4 = dyn_func(t+h,   state + h*k3,   accel, prm);
    state_new = state + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
