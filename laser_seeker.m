function [los_rate, los_vec, in_fov, seeker] = laser_seeker(state, target_pos, seeker, dt, params)
% LASER_SEEKER  Laser seeker model with plug-in filter selector.
%
%   Computes raw LOS measurement, checks FOV, then routes through the
%   selected filter algorithm to estimate los_vec and los_rate.
%
%   Filter selected via params.filter_type:
%     'simple'    - first-order low-pass (original baseline)
%     'alphabeta' - Alpha-Beta-Gamma fixed-gain tracker
%     'std'       - Super-Twisting Differentiator (HOSM)
%     'ckf'       - Cubature Kalman Filter + Singer model
%     'vbaekf'    - Variational Bayes Adaptive EKF (default/recommended)
%
%   All filter state is stored in seeker.filt_state (struct).

    uav_pos = state(1:3);

    % ===== 1. RAW LOS MEASUREMENT ====================================
    r_vec = target_pos - uav_pos;
    range = norm(r_vec);
    if range < 0.1, range = 0.1; end
    los_true = r_vec / range;

    % Add angle noise (Gaussian, applied as small LOS perturbation)
    noise_rad = deg2rad(params.noise_std_deg);
    perturb   = noise_rad * randn(3,1);
    % Keep perturbation orthogonal to LOS (angular noise, not range noise)
    perturb   = perturb - dot(perturb, los_true) * los_true;
    los_noisy = los_true + perturb;
    los_noisy = los_noisy / norm(los_noisy);

    % ===== 2. FOV CHECK (nadir-based, ground-attack seeker) ==========
    boresight = [0; 0; 1];   % NED nadir (z downward)
    dot_val   = dot(boresight, los_noisy);
    dot_val   = max(-1, min(1, dot_val));
    look_ang  = acos(dot_val);
    fov_rad   = deg2rad(params.fov_half_deg);
    in_fov    = (look_ang <= fov_rad);

    if ~in_fov
        los_rate = zeros(3,1);
        los_vec  = los_noisy;
        % Do NOT update filter state when outside FOV
        return
    end

    % ===== 3. FILTER DISPATCH ========================================
    ftype = params.filter_type;

    switch lower(ftype)
        % ---- simple first-order low-pass (original) -----------------
        case 'simple'
            if isempty(seeker.los_vec_prev)
                raw_rate = zeros(3,1);
            else
                raw_rate = (los_noisy - seeker.los_vec_prev) / dt;
            end
            alpha    = params.tau_filter / (params.tau_filter + dt);
            if isempty(seeker.los_rate_filt)
                filt_rate = raw_rate;
            else
                filt_rate = alpha * seeker.los_rate_filt + (1-alpha) * raw_rate;
            end
            los_vec  = los_noisy;
            los_rate = filt_rate;
            seeker.los_vec_prev  = los_noisy;
            seeker.los_rate_filt = filt_rate;

        % ---- Alpha-Beta-Gamma ---------------------------------------
        case 'alphabeta'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_alphabeta(los_noisy, seeker.filt_state, dt, params);

        % ---- Super-Twisting Differentiator --------------------------
        case 'std'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_std(los_noisy, seeker.filt_state, dt, params);

        % ---- Cubature Kalman Filter + Singer ------------------------
        case 'ckf'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_ckf(los_noisy, seeker.filt_state, dt, params);

        % ---- VB Adaptive EKF ----------------------------------------
        case 'vbaekf'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_vbaekf(los_noisy, seeker.filt_state, dt, params);

        % ---- Max Correntropy Criterion KF (post-2020) ---------------
        case 'mcc'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_mcc(los_noisy, seeker.filt_state, dt, params);

        % ---- Student-t Robust KF (post-2020) ------------------------
        case 'studt'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_studt(los_noisy, seeker.filt_state, dt, params);

        % ---- Fixed-Time Differentiator (post-2020) ------------------
        case 'ftd'
            [los_vec, los_rate, ~, seeker.filt_state] = ...
                seeker_ftd(los_noisy, seeker.filt_state, dt, params);

        otherwise
            error('laser_seeker: unknown filter_type "%s"', ftype);
    end

    seeker.range = range;
end
