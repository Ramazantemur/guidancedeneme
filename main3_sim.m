%% MAIN3_SIM.M — Seeker Algorithm Comparison (8 Algorithms × 3 Noise Scenarios)
%
%  Compares eight seeker/filter algorithms for UAV guidance LOS estimation
%  with a mid-flight laser target correction:
%
%  ┌──── Group A: Angle-domain filters ─────────────────────────────────────
%  │  Alg 1  'alpha'      Sliding-window mean of uLOS (pure signal processing)
%  │  Alg 2  'ekflos'     EKF: [el,ėl,az,ȧz,b_el,b_az]  measurement=uLOS
%  │  Alg 3  'ekfimu'     Same EKF + IMU bias states (8); accurate IMU rate
%  │  Alg 5  'ekfprop'    EKF-LOS but rate states set directly from IMU
%  │                       (no bias estimation — pure IMU propagation)
%  │  Alg 7  'imm'        Interacting Multiple Model EKF (slow + fast noise)
%  │  Alg 8  'riekf'      Robust-Iterated EKF — Huber M-estimator (IRLS)
%  └─────────────────────────────────────────────────────────────────────────
%  ┌──── Group B: Relative-position filters ─────────────────────────────────
%  │  Alg 4  'ekfrelpos'  EKF: [dr(3),dv(3)]; nonlinear h=uLOS; outputs target pos
%  │  Alg 6  'ukfrelpos'  UKF version of Alg 4 (sigma-point measurement)
%  └─────────────────────────────────────────────────────────────────────────
%
%  Mid-flight laser fix: at t_laser the true target shifts by delta_target.
%  The seeker must detect, track and guide to the new target.
%
%  Noise scenarios:
%    1. Nominal Gaussian (0.1°)
%    2. High Gaussian (1.5°)
%    3. Impulsive (0.1° + 10° spikes at 5%)
%
% =========================================================================
clear; clc; close all;
rng(42);

% =========================================================================
%% BASE PARAMETERS
% =========================================================================
base.g        = 9.81;
base.V_ref    = 150;
base.cx = 0; base.cy = 0; base.R = 1000; base.h_ref = 1500;
base.n_laps   = 2;
base.k_lat    = 0.5; base.k_alt = 4.0;
base.target_pos   = [0; 0; 0];
base.fov_half_deg = 45;
base.tau_filter   = 0.1;
base.N = 4; base.bias = [0;0;0]; base.a_max = 10*base.g;
base.dt = 0.05; base.t_max_guidance = 60; base.miss_threshold = 5;

% Mid-flight laser designation
base.t_laser        = 10.0;
base.delta_target   = [3.0; -4.0; 0.0];
base.new_target_pos = base.target_pos + base.delta_target;

% =========================================================================
%% ALGORITHM PARAMETERS
% =========================================================================

% -- Alg 1: Alpha sliding window ------------------------------------------
base.alpha_window = 10;

% -- Alg 2: EKF-LOS -------------------------------------------------------
base.ekflos_R0       = (deg2rad(0.1))^2;
base.ekflos_q_rate   = (deg2rad(0.5))^2;
base.ekflos_q_bias   = (deg2rad(0.01))^2;
base.ekflos_p0_ang   = deg2rad(3.0);
base.ekflos_p0_rate  = deg2rad(5.0);
base.ekflos_p0_bias  = deg2rad(1.0);

% -- Alg 3: EKF-IMU (angle-domain + bias) ---------------------------------
base.ekfimu_R0          = (deg2rad(0.1))^2;
base.ekfimu_q_rate      = (deg2rad(0.1))^2;
base.ekfimu_q_bias      = (deg2rad(0.01))^2;
base.ekfimu_q_imu_bias  = (deg2rad(0.005))^2;
base.ekfimu_imu_noise   = deg2rad(0.05);
base.ekfimu_p0_ang      = deg2rad(3.0);
base.ekfimu_p0_rate     = deg2rad(5.0);
base.ekfimu_p0_bias     = deg2rad(1.0);
base.ekfimu_p0_imu_bias = deg2rad(0.02);

% -- Alg 4: EKF-RelPos (relative position) --------------------------------
base.ekfrp_R0        = (deg2rad(0.15))^2;
base.ekfrp_q_pos     = 0.5;
base.ekfrp_q_vel     = 10.0;
base.ekfrp_p0_pos    = 300;
base.ekfrp_p0_vel    = 30;
base.ekfrp_use_alt   = true;

% -- Alg 5: EKF-IMU-Prop (direct IMU rate propagation, 6-state) ----------
base.ekfprop_R0        = (deg2rad(0.1))^2;
base.ekfprop_imu_noise = deg2rad(0.05);     % same IMU model as Alg 3
base.ekfprop_q_bias    = (deg2rad(0.01))^2;
base.ekfprop_p0_ang    = deg2rad(3.0);
base.ekfprop_p0_rate   = deg2rad(5.0);
base.ekfprop_p0_bias   = deg2rad(1.0);

% -- Alg 6: UKF-RelPos (same state as Alg 4, sigma-point meas update) ----
base.ekfukf_R0       = (deg2rad(0.15))^2;
base.ekfukf_q_pos    = 0.5;
base.ekfukf_q_vel    = 10.0;
base.ekfukf_p0_pos   = 300;
base.ekfukf_p0_vel   = 30;
base.ekfukf_use_alt  = true;

% -- Alg 7: IMM-EKF (two-model: slow / fast) ------------------------------
base.ekfimm_R0      = (deg2rad(0.1))^2;
base.ekfimm_q_slow  = (deg2rad(0.3))^2;    % low Q — slow manoeuvre model
base.ekfimm_q_fast  = (deg2rad(3.0))^2;    % high Q — fast / sudden change
base.ekfimm_q_bias  = (deg2rad(0.01))^2;
base.ekfimm_p_stay  = 0.90;                % Markov: prob of staying in same model
base.ekfimm_p0_ang  = deg2rad(3.0);
base.ekfimm_p0_rate = deg2rad(5.0);
base.ekfimm_p0_bias = deg2rad(1.0);

% -- Alg 8: RIEKF-Huber (robust iterated EKF) ----------------------------
base.riekf_R0      = (deg2rad(0.1))^2;
base.riekf_q_rate  = (deg2rad(0.5))^2;
base.riekf_q_bias  = (deg2rad(0.01))^2;
base.riekf_delta   = 1.5;    % Huber breakpoint [sigma units]; 1.345 = 95% Gaussian efficiency
base.riekf_n_iter  = 4;      % IRLS iterations per step
base.riekf_p0_ang  = deg2rad(3.0);
base.riekf_p0_rate = deg2rad(5.0);
base.riekf_p0_bias = deg2rad(1.0);

% -- Alg 9: EKF-3D (linear KF; states = [uLOS(3), uLOS_rate(3), bias(3)]) -
base.ekf3d_R0        = (deg2rad(0.1))^2;    % laser noise per axis
base.ekf3d_q_rate    = (deg2rad(0.5))^2;    % l_dot process noise PSD
base.ekf3d_q_bias    = (deg2rad(0.01))^2;   % bias random-walk PSD
base.ekf3d_p0_pos    = deg2rad(3.0);        % initial l  sigma
base.ekf3d_p0_rate   = deg2rad(5.0);        % initial l_dot sigma
base.ekf3d_p0_bias   = deg2rad(1.0);        % initial bias sigma

% -- Alg 10: EKF-3D-IMU (same 9-state; IMU rate as 2nd measurement) ------
base.ekf3dimu_R0           = (deg2rad(0.1))^2;  % laser noise per axis
base.ekf3dimu_R_imu        = (deg2rad(0.1))^2;  % IMU rate noise var per axis (rad/s)^2
base.ekf3dimu_q_rate       = (deg2rad(0.3))^2;  % tighter — IMU aids rate directly
base.ekf3dimu_q_bias       = (deg2rad(0.01))^2;
base.ekf3dimu_p0_pos       = deg2rad(3.0);
base.ekf3dimu_p0_rate      = deg2rad(5.0);
base.ekf3dimu_p0_bias      = deg2rad(1.0);
base.ekf3dimu_imu_noise_3d = deg2rad(0.05);     % 3D IMU rate sigma (rad/s per axis)

% =========================================================================
%% SCENARIOS
% =========================================================================
scenarios = {
    'Nominal (0.1°)',        0.1,  'gaussian',  0.0,   0.00;
    'High Gaussian (1.5°)',  1.5,  'gaussian',  0.0,   0.00;
    'Impulsive (0.1°+10°)',  0.1,  'impulsive', 10.0,  0.05;
};

% Algorithm list
filters = {'alpha', 'ekflos', 'ekfimu', 'ekfrelpos', ...
           'ekfprop', 'ukfrelpos', 'imm', 'riekf', ...
           'ekf3d', 'ekf3dimu'};
filter_labels = {'Alg1:  Alpha (Avg)', ...
                 'Alg2:  EKF-LOS', ...
                 'Alg3:  EKF-IMU (bias)', ...
                 'Alg4:  EKF-RelPos', ...
                 'Alg5:  EKF-Prop (IMU)', ...
                 'Alg6:  UKF-RelPos', ...
                 'Alg7:  IMM-EKF', ...
                 'Alg8:  RIEKF-Huber', ...
                 'Alg9:  EKF-3D (lin)', ...
                 'Alg10: EKF-3D+IMU'};

n_scen = size(scenarios,1);
n_filt = numel(filters);
miss_dist  = nan(n_filt, n_scen);
los_rmse   = nan(n_filt, n_scen);
max_accel  = nan(n_filt, n_scen);
target_err = nan(n_filt, n_scen);
success    = false(n_filt, n_scen);

% =========================================================================
%% PRE-COMPUTE CIRCULAR ORBIT HANDOVER
% =========================================================================
fprintf('Pre-computing circular orbit phase...\n');
p = base; p.filter_type = 'alpha'; p.noise_std_deg = 0.1;
x0 = p.cx + p.R; y0 = p.cy; z0 = -p.h_ref;
state = [x0; y0; z0; 0; p.V_ref; 0];
period = 2*pi*p.R/p.V_ref;
n_circ = ceil(p.n_laps*period/p.dt);
for k = 1:n_circ
    ac = circular_autopilot(state, p);
    state = rk4_step(@uav_dynamics, 0, state, ac, p, p.dt);
    V = state(4:6); state(4:6) = V*(p.V_ref/max(norm(V),1e-3));
end
state_handover = state;
fprintf('  Done. Handover: (%.0f, %.0f, %.0f) m\n\n', state_handover(1:3));

traj_store = cell(n_filt, 1);

% =========================================================================
%% MAIN COMPARISON LOOP
% =========================================================================
for s = 1:n_scen
    noise_deg   = scenarios{s,2};
    noise_type  = scenarios{s,3};
    impulse_mag = scenarios{s,4};
    impulse_p   = scenarios{s,5};
    fprintf('=== Scenario %d: %s ===\n', s, scenarios{s,1});

    for f = 1:n_filt
        ftype = filters{f};
        p = base;
        p.filter_type   = ftype;
        p.noise_std_deg = noise_deg;

        % Scale measurement noise params with scenario noise
        nd2 = (deg2rad(noise_deg))^2;
        p.ekflos_R0   = nd2;
        p.ekfimu_R0   = nd2;
        p.ekfrp_R0    = max(nd2, (deg2rad(0.15))^2);
        p.ekfprop_R0  = nd2;
        p.ekfukf_R0   = max(nd2, (deg2rad(0.15))^2);
        p.ekfimm_R0   = nd2;
        p.riekf_R0    = nd2;
        p.ekf3d_R0    = nd2;
        p.ekf3dimu_R0 = nd2;
        p.ekfukf_use_alt = base.ekfukf_use_alt;
        p.ekfrp_use_alt  = base.ekfrp_use_alt;

        rng(42 + f + s*100);

        state  = state_handover;
        seeker = init_seeker3(ftype, p);

        n_steps = ceil(p.t_max_guidance / p.dt);
        t_arr   = zeros(n_steps+1, 1);
        x_arr   = zeros(n_steps+1, 3);
        lr_arr  = zeros(n_steps+1, 3);
        ac_arr  = zeros(n_steps+1, 3);
        te_arr  = zeros(n_steps+1, 3);
        x_arr(1,:) = state(1:3)';

        current_target = p.target_pos;
        laser_fired    = false;
        k_end = n_steps; hit = false;

        for k = 1:n_steps
            t_now = k * p.dt;

            % ---- Laser fix at t_laser ----------------------------------------
            if ~laser_fired && t_now >= p.t_laser
                current_target = p.new_target_pos;
                laser_fired    = true;
                fprintf('    * [%s] Laser fix t=%.1fs → [%.1f,%.1f,%.1f]m\n', ...
                    ftype, t_now, current_target(1), current_target(2), current_target(3));
                seeker = reset_seeker3(seeker, ftype, p, current_target, ...
                                       state(1:3), state(4:6));
            end

            % ---- Noise level this step ---------------------------------------
            p.noise_std_deg = gen_noise3_deg(noise_deg, noise_type, impulse_mag, impulse_p);

            % ---- Seeker + filter ---------------------------------------------
            [los_rate, los_vec, in_fov, seeker, target_pos_est] = ...
                seeker_dispatch(state, current_target, seeker, p.dt, p, ftype);

            % ---- Guidance (BPNG) ---------------------------------------------
            if in_fov
                accel = bpng_guidance(state, target_pos_est, los_rate, p);
            else
                accel = zeros(3,1);
            end

            % ---- Dynamics ----------------------------------------------------
            state = rk4_step(@uav_dynamics, 0, state, accel, p, p.dt);
            V = state(4:6); state(4:6) = V*(p.V_ref/max(norm(V),1e-3));

            t_arr(k+1)    = t_now;
            x_arr(k+1,:) = state(1:3)';
            lr_arr(k+1,:) = los_rate';
            ac_arr(k+1,:) = accel';
            te_arr(k+1,:) = target_pos_est';

            if norm(state(1:3) - current_target) < p.miss_threshold || state(3) >= 0
                k_end = k+1; hit = true; break;
            end
        end

        md            = norm(state(1:3) - current_target);
        miss_dist(f,s) = md;
        success(f,s)  = hit && (md < 20);
        max_accel(f,s) = max(vecnorm(ac_arr(1:k_end,:),2,2)) / p.g;
        lr_trim        = lr_arr(2:k_end,:);
        los_rmse(f,s)  = mean(vecnorm(lr_trim,2,2));

        % Target estimate error (RelPos filters only)
        if ismember(ftype, {'ekfrelpos','ukfrelpos'})
            te_trim = te_arr(2:k_end,:);
            errs = vecnorm(te_trim - repmat(current_target', size(te_trim,1),1), 2, 2);
            target_err(f,s) = mean(errs);
        end

        if s == 1
            traj_store{f}.t      = t_arr(1:k_end);
            traj_store{f}.xyz    = x_arr(1:k_end,:);
            traj_store{f}.lr     = lr_arr(1:k_end,:);
            traj_store{f}.te     = te_arr(1:k_end,:);
            traj_store{f}.miss   = md;
            traj_store{f}.target = current_target;
        end

        tag = 'MISS'; if success(f,s), tag='HIT'; end
        fprintf('  [%-10s] %-28s : miss=%6.2fm  acc=%5.1fg  [%s]\n', ...
            ftype, filter_labels{f}, md, max_accel(f,s), tag);
    end
    fprintf('\n');
    p.noise_std_deg = noise_deg;
end

% =========================================================================
%% PRINT SUMMARY TABLE
% =========================================================================
fprintf('\n%s\n', repmat('=',1,110));
fprintf('  SEEKER COMPARISON — Miss Distance (m)  [OK = hit within 20m]\n');
fprintf('%s\n', repmat('=',1,110));
hdr = sprintf('%-30s', 'Algorithm');
for s = 1:n_scen
    lbl = scenarios{s,1}; if length(lbl)>20, lbl=lbl(1:20); end
    hdr = [hdr, sprintf('  %-22s',lbl)]; %#ok
end
fprintf('%s\n%s\n', hdr, repmat('-',1,110));
for f = 1:n_filt
    row = sprintf('%-30s', filter_labels{f});
    for s = 1:n_scen
        if success(f,s), cs = sprintf('%.1fm OK',miss_dist(f,s));
        else,            cs = sprintf('%.0fm --',miss_dist(f,s)); end
        row = [row, sprintf('  %-22s',cs)]; %#ok
    end
    fprintf('%s\n',row);
end
fprintf('%s\n', repmat('=',1,110));

fprintf('\n  Mean |LOS rate| (deg/s) — lower = smoother\n%s\n',repmat('-',1,110));
for f = 1:n_filt
    row = sprintf('%-30s', filter_labels{f});
    for s = 1:n_scen
        row = [row, sprintf('  %-22s',sprintf('%.4f',rad2deg(los_rmse(f,s))))]; %#ok
    end
    fprintf('%s\n',row);
end

fprintf('\n  Max Commanded Accel (g)\n%s\n',repmat('-',1,110));
for f = 1:n_filt
    row = sprintf('%-30s', filter_labels{f});
    for s = 1:n_scen
        row = [row, sprintf('  %-22s',sprintf('%.2fg',max_accel(f,s)))]; %#ok
    end
    fprintf('%s\n',row);
end
fprintf('%s\n\n', repmat('=',1,110));

% =========================================================================
%% PLOTS
% =========================================================================
colors_full = lines(n_filt);            % one distinct colour per algorithm
ls_map = {'-','--','-.',  ':', ...
          '-','--','-.',  ':', ...
          '-','--'};                    % 10 line styles
lw_map = [2.0, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8];  % 10 linewidths

% Mark which are RelPos-type (Alg 4, 6) vs angle-type (rest)
is_relpos = false(1,n_filt);
for ff = 1:n_filt
    if ismember(filters{ff}, {'ekfrelpos','ukfrelpos'}), is_relpos(ff) = true; end
end

% ---- 3D Trajectories (Scenario 1 — Nominal) ------------
figure('Name','3D Trajectories — All 8 Algorithms','Color','w','Position',[30 30 1100 700]);
hold on; grid on;
for f = 1:n_filt
    if isempty(traj_store{f}), continue; end
    xyz = traj_store{f}.xyz;
    plot3(xyz(:,2), xyz(:,1), -xyz(:,3), 'Color', colors_full(f,:), ...
          'LineStyle', ls_map{f}, 'LineWidth', lw_map(f), ...
          'DisplayName', sprintf('%s (%.1fm)', filter_labels{f}, traj_store{f}.miss));
end
tp  = base.target_pos;
ntp = base.new_target_pos;
plot3(tp(2),  tp(1),  -tp(3),  'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.7 0.7 0.7], ...
      'DisplayName', 'Nominal Target');
plot3(ntp(2), ntp(1), -ntp(3), 'r*', 'MarkerSize', 14, 'LineWidth', 2.5, ...
      'DisplayName', 'Laser Target');
xlabel('East (m)'); ylabel('North (m)'); zlabel('Altitude (m)');
title(sprintf('BPNG Guidance — %d Seeker Algorithms (Nominal Noise, Laser Fix t=%.0fs)', n_filt, base.t_laser));
legend('Location','northeast','FontSize',7,'NumColumns',2); view(30,30);

% ---- LOS Rate Magnitude (Scenario 1) ---
figure('Name','LOS Rate — All Algorithms','Color','w','Position',[30 80 1100 400]);
hold on; grid on;
for f = 1:n_filt
    if isempty(traj_store{f}), continue; end
    lr = traj_store{f}.lr; t = traj_store{f}.t;
    plot(t, rad2deg(vecnorm(lr,2,2)), 'Color', colors_full(f,:), ...
         'LineStyle', ls_map{f}, 'LineWidth', lw_map(f), 'DisplayName', filter_labels{f});
end
xline(base.t_laser,'k--','LineWidth',1.5,'DisplayName','Laser Fix');
xlabel('Time (s)'); ylabel('|\omega_{LOS}|  (deg/s)');
title(sprintf('LOS Rate Estimate — All %d Algorithms (Nominal Noise)', n_filt));
legend('Location','northeast','FontSize',7,'NumColumns',2); ylim([0,20]);

% ---- Miss Distance Bar Chart (all scenarios) ---
figure('Name',sprintf('Miss Distance — All %d Algorithms', n_filt),'Color','w','Position',[30 130 1100 500]);
b = bar(miss_dist','grouped');
for f = 1:n_filt, b(f).FaceColor = colors_full(f,:); end
set(gca,'XTickLabel',{scenarios{:,1}},'FontSize',9);
ylabel('Miss Distance (m)');
title(sprintf('Miss Distance — %d Algorithms \times 3 Noise Scenarios', n_filt));
legend(filter_labels,'Location','northwest','FontSize',7,'NumColumns',2);
grid on; yline(base.miss_threshold,'r--','LineWidth',2);

% ---- Radar/Spider chart: normalised performance summary ------------------
figure('Name','Performance Radar Chart','Color','w','Position',[30 180 700 600]);
categories = {'Nominal\nMiss','High Gaussian\nMiss','Impulsive\nMiss',...
               'Nominal\nLOS RMSE','LOS RMSE\n(Impulsive)','Peak\nAccel (Nominal)'};
n_cat = 6;
% Collect per-algorithm score (0=worst, 1=best, higher=better)
raw = [miss_dist(:,1), miss_dist(:,2), miss_dist(:,3), ...
       rad2deg(los_rmse(:,1)), rad2deg(los_rmse(:,3)), max_accel(:,1)];
% Normalise: lower raw value → higher score (1 = best, 0 = worst)
scores = zeros(n_filt, n_cat);
for c = 1:n_cat
    col = raw(:,c);
    col(isnan(col)) = max(col(~isnan(col)));
    scores(:,c) = 1 - (col - min(col)) / (max(col) - min(col) + eps);
end
% Spider plot using polar axes (manual implementation)
theta = linspace(0, 2*pi, n_cat+1);
theta(end) = theta(1);
ax = axes; hold on; axis off;
title('Normalised Performance Radar (higher = better)', 'FontSize', 11);
% Draw grid circles
for r = 0.25:0.25:1.0
    x_ring = r * cos(theta(1:end-1));
    y_ring = r * sin(theta(1:end-1));
    fill([x_ring, x_ring(1)], [y_ring, y_ring(1)], ...
         'w', 'EdgeColor', [0.8 0.8 0.8], 'LineStyle', '--');
end
% Draw spokes and category labels
for c = 1:n_cat
    plot([0, 1.1*cos(theta(c))], [0, 1.1*sin(theta(c))], 'k:', 'LineWidth', 0.7);
    lbl = strsplit(categories{c}, '\n');
    text(1.2*cos(theta(c)), 1.2*sin(theta(c)), lbl, ...
         'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end
% Plot each algorithm
for f = 1:n_filt
    vals = [scores(f,:), scores(f,1)];   % close the polygon
    xv   = vals .* cos(theta);
    yv   = vals .* sin(theta);
    plot(xv, yv, 'Color', colors_full(f,:), 'LineStyle', ls_map{f}, ...
         'LineWidth', 1.5, 'DisplayName', filter_labels{f});
end
legend('Location','southoutside','FontSize',7,'NumColumns',2,'Orientation','horizontal');
xlim([-1.5 1.5]); ylim([-1.5 1.5]); axis equal;

% ---- Target estimate: Alg 4 vs Alg 6 side by side (Scenario 1) --------
figure('Name','Target Position Estimate — Alg4 vs Alg6','Color','w','Position',[30 220 1100 450]);
relpos_algs = find(is_relpos);
n_rp = numel(relpos_algs);
for rp = 1:n_rp
    f   = relpos_algs(rp);
    subplot(n_rp, 2, (rp-1)*2 + 1); hold on; grid on;
    te  = traj_store{f}.te;
    t   = traj_store{f}.t;
    tgt = traj_store{f}.target;
    plot(t, te(:,1), 'b-', t, te(:,2), 'r-', t, te(:,3), 'k-', 'LineWidth',1.5);
    yline(tgt(1),'b--'); yline(tgt(2),'r--'); yline(tgt(3),'k--');
    xline(base.t_laser,'m--','LineWidth',1.5);
    ylabel('m'); title(sprintf('%s — Estimate vs True', filter_labels{f}),'FontSize',8);
    legend('N','E','D','Location','northeast','FontSize',7);

    subplot(n_rp, 2, (rp-1)*2 + 2); hold on; grid on;
    errs = vecnorm(te - repmat(tgt', size(te,1),1), 2, 2);
    plot(t, errs, 'Color', colors_full(f,:), 'LineWidth', 2);
    xline(base.t_laser,'k--','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Error (m)');
    title(sprintf('%s — Position Error', filter_labels{f}),'FontSize',8);
    grid on;
end

fprintf('All plots generated. Simulation complete.\n');

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function seeker = init_seeker3(ftype, params)
    seeker.los_vec_prev  = [];
    seeker.los_rate_filt = [];
    seeker.range         = 0;
    seeker.target_est    = params.target_pos;
    switch lower(ftype)
        case 'alpha'
            seeker.filt_state.buf   = [];
            seeker.filt_state.ptr   = 1;
            seeker.filt_state.count = 0;
            seeker.filt_state.prev  = [];
        case 'imm'
            seeker.filt_state.x      = [];
            seeker.filt_state.P      = [];
            seeker.filt_state.R      = [];
            seeker.filt_state.mu     = [];
            seeker.filt_state.trans  = [];
            seeker.filt_state.models = [];
        otherwise   % all EKF / UKF variants (incl. ekf3d, ekf3dimu)
            seeker.filt_state.x        = [];
            seeker.filt_state.P        = [];
            seeker.filt_state.R        = [];
            seeker.filt_state.R1       = [];
            seeker.filt_state.R2       = [];
            seeker.filt_state.vel_prev = [];
    end
end

function seeker = reset_seeker3(seeker, ftype, ~, new_target, ~, ~)
    % Reset filter state on laser fix (forces re-initialisation)
    seeker.target_est = new_target;
    switch lower(ftype)
        case 'alpha'
            seeker.filt_state.buf   = [];
            seeker.filt_state.ptr   = 1;
            seeker.filt_state.count = 0;
            seeker.filt_state.prev  = [];
        case 'imm'
            seeker.filt_state.x      = [];
            seeker.filt_state.mu     = [];
            seeker.filt_state.models = [];
        otherwise
            seeker.filt_state.x = [];
            seeker.filt_state.P = [];
    end
end

function [los_rate, los_vec, in_fov, seeker, target_pos_est] = ...
         seeker_dispatch(state, target_pos, seeker, dt, params, ftype)

    uav_pos = state(1:3);
    uav_vel = state(4:6);

    % Raw noisy LOS measurement
    r_vec  = target_pos - uav_pos;
    range  = norm(r_vec);
    if range < 0.1, range = 0.1; end
    los_true = r_vec / range;
    noise_rad = deg2rad(params.noise_std_deg);
    perturb   = noise_rad * randn(3,1);
    perturb   = perturb - dot(perturb, los_true) * los_true;
    los_noisy = los_true + perturb;
    los_noisy = los_noisy / norm(los_noisy);

    % FOV check
    boresight = [0;0;1];
    in_fov = (acos(max(-1,min(1,dot(boresight,los_noisy)))) <= deg2rad(params.fov_half_deg));

    seeker.range = range;
    target_pos_est = target_pos;   % default (Alg 1,2,3,5,7,8)

    if ~in_fov
        los_rate = zeros(3,1); los_vec = los_noisy; return
    end

    switch lower(ftype)
        case 'alpha'
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_alpha(los_noisy, seeker.filt_state, dt, params);

        case 'ekflos'
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_ekf_los(los_noisy, seeker.filt_state, dt, params);

        case 'ekfimu'
            imu_rate = compute_imu_rate(uav_pos, uav_vel, target_pos, params);
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_ekf_imu(los_noisy, imu_rate, seeker.filt_state, dt, params);

        case 'ekfrelpos'
            [los_vec, los_rate, target_pos_est, seeker.filt_state] = ...
                seeker_ekf_relpos(los_noisy, uav_pos, uav_vel, seeker.filt_state, dt, params);

        case 'ekfprop'
            imu_rate = compute_imu_rate(uav_pos, uav_vel, target_pos, params);
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_ekf_imu_prop(los_noisy, imu_rate, seeker.filt_state, dt, params);

        case 'ukfrelpos'
            [los_vec, los_rate, target_pos_est, seeker.filt_state] = ...
                seeker_ukf_relpos(los_noisy, uav_pos, uav_vel, seeker.filt_state, dt, params);

        case 'imm'
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_imm_ekf(los_noisy, seeker.filt_state, dt, params);

        case 'riekf'
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_riekf(los_noisy, seeker.filt_state, dt, params);

        case 'ekf3d'
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_ekf_3d(los_noisy, seeker.filt_state, dt, params);

        case 'ekf3dimu'
            % Compute 3D IMU rate (dl/dt in NED, rad/s per axis)
            imu_rate_3d = compute_imu_rate_3d(uav_pos, uav_vel, target_pos, params);
            [los_vec, los_rate, seeker.filt_state] = ...
                seeker_ekf_3d_imu(los_noisy, imu_rate_3d, seeker.filt_state, dt, params);

        otherwise
            error('seeker_dispatch: unknown filter "%s"', ftype);
    end
end

% ---------- IMU LOS rate computation (used by Alg 3 and Alg 5) ----------
function imu_rate = compute_imu_rate(uav_pos, uav_vel, target_pos, params)
    r_vec   = target_pos - uav_pos;
    r       = norm(r_vec);
    if r < 0.1, r = 0.1; end
    los_hat = r_vec / r;
    v_rel   = -uav_vel;

    % CORRECT formula: omega = cross(los_hat, v_rel) / r
    omega_vec = cross(los_hat, v_rel) / r;

    % Project onto orthonormal (el, az) tangent basis
    [az_t, el_t] = los_to_angles_local(los_hat);
    e_el      = [-sin(el_t)*cos(az_t); -sin(el_t)*sin(az_t); -cos(el_t)];
    e_az_unit = [-sin(az_t);            cos(az_t);             0         ];

    el_dot_true = dot(omega_vec, e_el);
    cos_el = cos(el_t);
    if abs(cos_el) > 1e-3
        az_dot_true = dot(omega_vec, e_az_unit) / cos_el;
    else
        az_dot_true = 0;
    end

    imu_noise_sigma = params.ekfimu_imu_noise;   % reused for Alg 3/5
    imu_rate = [el_dot_true; az_dot_true] + imu_noise_sigma * randn(2,1);
end

% ---------- 3D IMU uLOS-rate computation (used by Alg 10) ----------------
function l_dot_imu = compute_imu_rate_3d(uav_pos, uav_vel, target_pos, params)
    % Returns 3D dl/dt = d(uLOS)/dt from kinematics + noise.
    % Formula: l_dot = (v_rel - (v_rel·l̂)·l̂) / r
    % where v_rel = -v_uav (stationary target)
    r_vec = target_pos - uav_pos;
    r     = norm(r_vec);
    if r < 0.1, r = 0.1; end
    los_hat  = r_vec / r;
    v_rel    = -uav_vel;   % stationary target assumption

    % 3D LOS rate (tangent to unit sphere, perpendicular to los_hat)
    l_dot_true = (v_rel - dot(v_rel, los_hat) * los_hat) / r;

    % Add 3D isotropic IMU noise
    sigma_3d  = params.ekf3dimu_imu_noise_3d;
    l_dot_imu = l_dot_true + sigma_3d * randn(3,1);
end

function [az, el] = los_to_angles_local(v)
    v = v / (norm(v) + eps);
    el = asin(max(-1, min(1, -v(3))));
    az = atan2(v(2), v(1));
end

% ---------- Noise generator ----------------------------------------------
function nd = gen_noise3_deg(sigma, noise_type, imp_mag, imp_prob)
    switch noise_type
        case 'gaussian';  nd = sigma;
        case 'impulsive'; nd = sigma; if rand() < imp_prob, nd = imp_mag; end
        case 'laplacian'; nd = sigma*(1+abs(randn())*0.5);
        otherwise;        nd = sigma;
    end
end

% ---------- RK4 integrator -----------------------------------------------
function state_new = rk4_step(dyn_func, t, state, accel, prm, h)
    k1 = dyn_func(t,     state,         accel, prm);
    k2 = dyn_func(t+h/2, state+h/2*k1, accel, prm);
    k3 = dyn_func(t+h/2, state+h/2*k2, accel, prm);
    k4 = dyn_func(t+h,   state+h*k3,   accel, prm);
    state_new = state + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end
