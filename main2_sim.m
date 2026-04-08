%% MAIN2_SIM.M — UAV Guidance + Post-2020 Filter Comparison
%
%  Tests 5 filters: 2 best from Phase 1 (CKF, VB-AEKF) as baselines,
%  plus 3 new post-2020 algorithms:
%    'ckf'    - Cubature Kalman Filter + Singer (2009, baseline)
%    'vbaekf' - Variational Bayes AEKF (2019, baseline)
%    'mcc'    - Max Correntropy Criterion KF (2021-2022)  [NEW]
%    'studt'  - Student-t Robust KF (2020)               [NEW]
%    'ftd'    - Fixed-Time Differentiator (2020-2021)    [NEW]
%
%  Scenarios — include NON-GAUSSIAN noise to expose robustness advantage:
%    1. Nominal Gaussian       0.1 deg sigma
%    2. High Gaussian          1.5 deg sigma
%    3. Impulsive (5% chance)  0.1 deg + 10 deg outliers
%    4. Mixed: Laplacian 1 deg (heavy-tailed continuous)
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
base.target_pos = [0;0;0];
base.fov_half_deg = 45;
base.tau_filter   = 0.1;        % for 'simple' only (not used in this file)
base.N = 4; base.bias = [0;0;0]; base.a_max = 10*base.g;
base.dt = 0.05; base.t_max_guidance = 60; base.miss_threshold = 3;

% ---- CKF (baseline) ---------------------------------------------------
base.ckf_sigma_m = 0.05; base.ckf_tau_m = 3.0;
base.ckf_R = (deg2rad(0.1))^2;

% ---- VB-AEKF (baseline) -----------------------------------------------
base.vb_tau_m = 3.0; base.vb_sigma_m = 0.15;
base.vb_R0 = (deg2rad(0.1))^2; base.vb_nu_R = 8; base.vb_nu_Q = 5;

% ---- MCCKF (new, 2021) ------------------------------------------------
base.mcc_sigma_m = 0.05; base.mcc_tau_m = 3.0;
base.mcc_R0      = (deg2rad(0.1))^2;
base.mcc_kernel  = deg2rad(0.3);   % MCC kernel bandwidth σ (rad)
base.mcc_iter    = 1;              % single-pass: stable + sufficient for outlier rejection

% ---- Student-t KF (new, 2020) -----------------------------------------
base.st_sigma_m = 0.05; base.st_tau_m = 3.0;
base.st_R0      = (deg2rad(0.1))^2;
base.st_nu      = 4;               % degrees of freedom (lower = heavier tail)
base.st_iter    = 5;               % EM iterations per step

% ---- Fixed-Time Differentiator (new, 2020-2021) -----------------------
base.ftd_Tmax   = 2.0;            % predefined convergence time (s)
base.ftd_n_sub  = 20;             % sub-steps per dt

% =========================================================================
%% SCENARIOS
% =========================================================================
% Noise type: 'gaussian', 'impulsive', 'laplacian'
scenarios = {
    'Nominal Gaussian (0.1°)',  0.1,  'gaussian',  0.00,  0;
    'High Gaussian (1.5°)',     1.5,  'gaussian',  0.00,  0;
    'Impulsive (0.1°+10° 5%)',  0.1,  'impulsive', 10.0,  0.05;
    'Laplacian (1.0° heavy)',   1.0,  'laplacian', 0.00,  0;
};
% Columns: label, sigma_deg, type, impulse_mag_deg, impulse_prob

filters = {'ckf', 'vbaekf', 'mcc', 'studt', 'ftd'};
filter_labels = {'CKF+Singer (2009)', 'VB-AEKF (2019)', ...
                 'MCCKF (2021-NEW)', 'Student-t KF (2020-NEW)', ...
                 'Fixed-Time Diff (2020-NEW)'};

n_scen = size(scenarios,1);
n_filt = numel(filters);
miss_dist = nan(n_filt,n_scen);
los_rmse  = nan(n_filt,n_scen);
max_accel = nan(n_filt,n_scen);
success   = false(n_filt,n_scen);

% =========================================================================
%% PRE-COMPUTE CIRCULAR PHASE
% =========================================================================
fprintf('Pre-computing circular orbit phase...\n');
p = base; p.filter_type = 'ckf'; p.noise_std_deg = 0.1;
x0 = p.cx + p.R; y0 = p.cy; z0 = -p.h_ref;
state = [x0; y0; z0; 0; p.V_ref; 0];
period = 2*pi*p.R/p.V_ref;
n_circ = ceil(p.n_laps*period/p.dt);
for k = 1:n_circ
    ac = circular_autopilot(state,p);
    state = rk4_step(@uav_dynamics,0,state,ac,p,p.dt);
    V=state(4:6); state(4:6)=V*(p.V_ref/max(norm(V),1e-3));
end
state_handover = state;
fprintf('  Done. Handover: (%.0f, %.0f, %.0f) m\n\n', state_handover(1:3));

traj_store = cell(n_filt,1);

% =========================================================================
%% COMPARISON LOOP
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
        % Update R0 for model-based filters
        p.ckf_R  = (deg2rad(noise_deg))^2;
        p.vb_R0  = (deg2rad(noise_deg))^2;
        p.mcc_R0 = (deg2rad(noise_deg))^2;
        p.mcc_kernel = deg2rad(max(noise_deg*1.5, 0.2));  % tune kernel to noise
        p.st_R0  = (deg2rad(noise_deg))^2;

        rng(42 + f + s*100);

        state  = state_handover;
        seeker = init_seeker(ftype);

        n_steps = ceil(p.t_max_guidance/p.dt);
        t_arr   = zeros(n_steps+1,1);
        x_arr   = zeros(n_steps+1,3);
        lr_arr  = zeros(n_steps+1,3);
        ac_arr  = zeros(n_steps+1,3);
        x_arr(1,:) = state(1:3)';

        k_end = n_steps; hit = false;
        for k = 1:n_steps
            % ---- Generate noisy LOS measurement --------------------
            p.noise_std_deg = gen_noise_deg(noise_deg, noise_type, ...
                                            impulse_mag, impulse_p);

            [los_rate, los_vec, in_fov, seeker] = ...
                laser_seeker(state, p.target_pos, seeker, p.dt, p);

            if in_fov
                accel = bpng_guidance(state, p.target_pos, los_rate, p);
            else
                accel = zeros(3,1);
            end

            state = rk4_step(@uav_dynamics,0,state,accel,p,p.dt);
            V=state(4:6); state(4:6)=V*(p.V_ref/max(norm(V),1e-3));

            t_arr(k+1)    = k*p.dt;
            x_arr(k+1,:)  = state(1:3)';
            lr_arr(k+1,:) = los_rate';
            ac_arr(k+1,:) = accel';

            if norm(state(1:3)-p.target_pos) < p.miss_threshold || state(3)>=0
                k_end=k+1; hit=true; break;
            end
        end

        md              = norm(state(1:3)-p.target_pos);
        miss_dist(f,s)  = md;
        success(f,s)    = hit && (md < 20);
        max_accel(f,s)  = max(vecnorm(ac_arr(1:k_end,:),2,2))/p.g;
        lr_trim         = lr_arr(2:k_end,:);
        los_rmse(f,s)   = mean(vecnorm(lr_trim,2,2));

        if s == 1
            traj_store{f}.t    = t_arr(1:k_end);
            traj_store{f}.xyz  = x_arr(1:k_end,:);
            traj_store{f}.lr   = lr_arr(1:k_end,:);
            traj_store{f}.miss = md;
        end

        tag = 'MISS'; if success(f,s), tag='HIT'; end
        fprintf('  [%-7s] %-22s : miss=%6.2f m  t=%5.1f s  [%s]\n', ...
            ftype, filter_labels{f}, md, t_arr(k_end), tag);
    end
    fprintf('\n');
    % Reset noise seed
    p.noise_std_deg = noise_deg;
end

% =========================================================================
%% PRINT TABLE
% =========================================================================
fprintf('\n%s\n', repmat('=',1,90));
fprintf('  POST-2020 FILTER COMPARISON — Miss Distance (m)\n');
fprintf('%s\n', repmat('=',1,90));
hdr = sprintf('%-26s', 'Filter');
for s=1:n_scen
    lbl = scenarios{s,1}; if length(lbl)>18, lbl=lbl(1:18); end
    hdr = [hdr, sprintf('  %-19s',lbl)]; %#ok
end
fprintf('%s\n', hdr);
fprintf('%s\n', repmat('-',1,90));
for f=1:n_filt
    row = sprintf('%-26s', filter_labels{f});
    for s=1:n_scen
        if success(f,s)
            cs = sprintf('%.1f m OK', miss_dist(f,s));
        else
            cs = sprintf('%.0f m --', miss_dist(f,s));
        end
        row=[row, sprintf('  %-19s',cs)]; %#ok
    end
    fprintf('%s\n',row);
end
fprintf('%s\n', repmat('=',1,90));

fprintf('\n  Mean LOS Rate Magnitude (rad/s) — lower = smoother\n');
fprintf('%s\n', repmat('-',1,90));
for f=1:n_filt
    row=sprintf('%-26s',filter_labels{f});
    for s=1:n_scen
        row=[row, sprintf('  %-19s',sprintf('%.4f',los_rmse(f,s)))]; %#ok
    end
    fprintf('%s\n',row);
end
fprintf('%s\n\n', repmat('=',1,90));

% =========================================================================
%% PLOTS
% =========================================================================
colors = lines(n_filt);
ls_map = {'-','--','-.', ':','--'};

figure('Name','3D Trajectories — Post-2020 Filters','Color','w','Position',[50 50 900 600]);
hold on; grid on;
for f=1:n_filt
    xyz=traj_store{f}.xyz;
    plot3(xyz(:,2),xyz(:,1),-xyz(:,3),'Color',colors(f,:),...
          'LineStyle',ls_map{f},'LineWidth',1.8,...
          'DisplayName',sprintf('%s (%.1fm)',filter_labels{f},traj_store{f}.miss));
end
tp=base.target_pos;
plot3(tp(2),tp(1),-tp(3),'kp','MarkerSize',14,'MarkerFaceColor','k','DisplayName','Target');
xlabel('East (m)'); ylabel('North (m)'); zlabel('Alt (m)');
title('BPNG Guidance Phase — Post-2020 Filters (Nominal Noise)');
legend('Location','northeast','FontSize',8); view(30,30);

figure('Name','Miss Distance — Post-2020 Comparison','Color','w','Position',[50 70 900 450]);
b=bar(miss_dist','grouped');
for f=1:n_filt, b(f).FaceColor=colors(f,:); end
set(gca,'XTickLabel',{scenarios{:,1}},'FontSize',8);
ylabel('Miss Distance (m)');
title('Miss Distance: Post-2020 Algorithms vs Baselines');
legend(filter_labels,'Location','northwest','FontSize',8);
grid on; yline(base.miss_threshold,'r--','3m Threshold');

figure('Name','LOS Rate — Post-2020','Color','w','Position',[50 90 900 400]);
hold on; grid on;
for f=1:n_filt
    lr=traj_store{f}.lr; t=traj_store{f}.t;
    plot(t,rad2deg(vecnorm(lr,2,2)),'Color',colors(f,:),...
         'LineStyle',ls_map{f},'LineWidth',1.5,'DisplayName',filter_labels{f});
end
xlabel('Time (s)'); ylabel('|omega-LOS| (deg/s)');
title('LOS Rate Estimate Magnitude — All Post-2020 Filters');
legend('Location','northeast','FontSize',8);

% =========================================================================
%% HELPERS
% =========================================================================
function nd = gen_noise_deg(sigma, noise_type, imp_mag, imp_prob)
    % Generate per-step noise level (returned as sigma for this timestep)
    switch noise_type
        case 'gaussian'
            nd = sigma;
        case 'impulsive'
            % With prob imp_prob, fire an outlier spike
            if rand() < imp_prob
                nd = imp_mag;   % occasional large spike
            else
                nd = sigma;
            end
        case 'laplacian'
            % Scale: Laplacian with std = sigma → b = sigma/sqrt(2)
            % Approximate by returning sigma*sqrt(2) for heavier draws
            % (actual Laplacian sampled in laser_seeker via randn rescale)
            nd = sigma * (1 + abs(randn()) * 0.5);  % rough Laplacian approx
        otherwise
            nd = sigma;
    end
end

function s = init_seeker(ftype)
    s.los_vec_prev  = [];
    s.los_rate_filt = [];
    s.range         = 0;
    switch lower(ftype)
        case 'alphabeta'
            s.filt_state.x_el = []; s.filt_state.x_az = [];
        case {'std','ftd'}
            s.filt_state.z_el     = []; s.filt_state.z_az     = [];
            s.filt_state.z1_el_sm = 0;  s.filt_state.z1_az_sm = 0;
        case {'ckf','vbaekf','mcc','studt'}
            s.filt_state.x = []; s.filt_state.P = [];
            s.filt_state.R = []; s.filt_state.Q = [];
            s.filt_state.k = 1;
        otherwise   % simple
    end
end

function state_new = rk4_step(dyn_func,t,state,accel,prm,h)
    k1=dyn_func(t,    state,        accel,prm);
    k2=dyn_func(t+h/2,state+h/2*k1,accel,prm);
    k3=dyn_func(t+h/2,state+h/2*k2,accel,prm);
    k4=dyn_func(t+h,  state+h*k3,  accel,prm);
    state_new=state+(h/6)*(k1+2*k2+2*k3+k4);
end
