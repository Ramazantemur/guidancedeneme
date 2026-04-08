function [los_vec, los_rate, filt] = seeker_imm_ekf(los_noisy, filt, dt, params)
% SEEKER_IMM_EKF — Algorithm 7: Interacting Multiple Model EKF
%
%   Uses the standard two-model IMM framework to adaptively blend two
%   EKF models with different process noise levels:
%
%     Model 1 (M1, "slow"): low process noise  — handles smooth LOS motion
%     Model 2 (M2, "fast"): high process noise — handles rapid manoeuvres
%                            and sudden target changes (laser fix events)
%
%   Both models share the same 6-state structure as Algorithm 2:
%     x = [el; el_dot; az; az_dot; el_bias; az_bias]
%
%   IMM Algorithm (per step):
%     1. Interaction/Mixing:
%          c_j   = Σ_i p_ij * μ_i         (normalising constant per model)
%          μ_ij  = p_ij * μ_i / c_j       (mixing weights)
%          x̂0_j  = Σ_i μ_ij * x̂_i        (mixed initial state for model j)
%          P0_j  = Σ_i μ_ij*(P_i + δ_i δ_i') where δ_i = x̂_i - x̂0_j
%     2. Mode-conditioned EKF:
%          Predict + update model j independently with its own Q_j
%     3. Likelihood calculation:
%          Λ_j = N(innov_j; 0, S_j)       (Gaussian likelihood)
%     4. Model weight update:
%          μ_j = Λ_j * c_j / Σ_k Λ_k c_k
%     5. Fusion:
%          x̂ = Σ_j μ_j * x̂_j
%          P  = Σ_j μ_j * (P_j + (x̂_j - x̂)(x̂_j - x̂)')
%
%   Why IMM excels here:
%     - During straight flight: M1 dominates → smooth, low-noise estimate
%     - During laser target update: innovation jumps → M2 dominates briefly
%     - After settling: M1 takes over again
%     This gives automatic, lag-free adaptation to target changes.
%
%   Markov transition matrix (high persistence):
%     P_trans = [p11 p12;   typically [0.95 0.05;
%               p21 p22]               0.05 0.95]
%
%   params:
%     .ekfimm_R0        - measurement noise var (rad^2)
%     .ekfimm_q_slow    - M1 (slow) rate process noise PSD
%     .ekfimm_q_fast    - M2 (fast) rate process noise PSD (>> q_slow)
%     .ekfimm_q_bias    - bias process noise PSD (both models)
%     .ekfimm_p_stay    - P(stay in same model), default 0.9
%     .ekfimm_p0_ang, .ekfimm_p0_rate, .ekfimm_p0_bias  (initial P)

    n_x = 6;
    n_z = 2;
    N   = 2;   % number of models

    % ---- Measurement: noisy LOS → angles ------------------------------------
    [az_m, el_m] = los_to_angles(los_noisy);
    z_k = [el_m; az_m];

    % ---- Initialise ----------------------------------------------------------
    if isempty(filt.x)
        p_stay = params.ekfimm_p_stay;
        filt.trans = [p_stay, 1-p_stay; 1-p_stay, p_stay];  % Markov matrix
        filt.mu    = [0.5; 0.5];   % equal initial model probabilities

        x0 = [el_m; 0; az_m; 0; 0; 0];
        P0 = diag([params.ekfimm_p0_ang,  params.ekfimm_p0_rate, ...
                   params.ekfimm_p0_ang,  params.ekfimm_p0_rate, ...
                   params.ekfimm_p0_bias, params.ekfimm_p0_bias].^2);
        R0 = params.ekfimm_R0 * eye(n_z);

        filt.models = cell(N,1);
        for j = 1:N
            filt.models{j}.x = x0;
            filt.models{j}.P = P0;
            filt.models{j}.R = R0;
        end
        filt.x = x0;   % fused state (for output)
        filt.P = P0;
    end

    % ---- F (same for both models) -------------------------------------------
    F = [1, dt,  0,  0,  0,  0;
         0,  1,  0,  0,  0,  0;
         0,  0,  1, dt,  0,  0;
         0,  0,  0,  1,  0,  0;
         0,  0,  0,  0,  1,  0;
         0,  0,  0,  0,  0,  1];

    H = [1, 0, 0, 0, 1, 0;
         0, 0, 1, 0, 0, 1];

    % Build Q for each model
    q_b   = params.ekfimm_q_bias;
    q_s   = params.ekfimm_q_slow;    % M1: slow manoeuvre
    q_f   = params.ekfimm_q_fast;    % M2: fast / large manoeuvre
    Q_mods = {diag([q_s*dt^3/3, q_s*dt, q_s*dt^3/3, q_s*dt, q_b*dt, q_b*dt]), ...
              diag([q_f*dt^3/3, q_f*dt, q_f*dt^3/3, q_f*dt, q_b*dt, q_b*dt])};

    % ========================================================
    % STEP 1: Interaction / Mixing
    % ========================================================
    % Predicted model probabilities: c_j = Σ_i p_ij * mu_i
    c = filt.trans' * filt.mu;   % (2×1) normalising constants

    % Mixing weights: mu_ij = p_ij * mu_i / c_j
    mu_mix = zeros(N, N);
    for j = 1:N
        for i = 1:N
            mu_mix(i,j) = filt.trans(i,j) * filt.mu(i) / (c(j) + eps);
        end
    end

    % Compute mixed initial state & covariance for each model
    x_mix = zeros(n_x, N);
    P_mix = cell(N,1);
    for j = 1:N
        x_mix(:,j) = zeros(n_x,1);
        for i = 1:N
            x_mix(:,j) = x_mix(:,j) + mu_mix(i,j) * filt.models{i}.x;
        end
        P_mix{j} = zeros(n_x,n_x);
        for i = 1:N
            delta = filt.models{i}.x - x_mix(:,j);
            P_mix{j} = P_mix{j} + mu_mix(i,j) * (filt.models{i}.P + delta*delta');
        end
        P_mix{j} = 0.5*(P_mix{j} + P_mix{j}');
    end

    % ========================================================
    % STEP 2: Mode-conditioned EKF for each model
    % ========================================================
    innov_j = zeros(n_z, N);
    S_j     = zeros(n_z, n_z, N);
    Lam     = zeros(N, 1);    % likelihoods

    for j = 1:N
        xj = x_mix(:,j);
        Pj = P_mix{j};
        Qj = Q_mods{j};
        Rj = filt.models{j}.R;

        % Predict
        xj_p = F * xj;
        Pj_p = F * Pj * F' + Qj;
        Pj_p = 0.5*(Pj_p + Pj_p');

        % Update
        z_pred = H * xj_p;
        innov  = z_k - z_pred;
        innov(2) = angle_wrap(innov(2));

        Sj = H * Pj_p * H' + Rj;
        Kj = Pj_p * H' / Sj;

        xj_up = xj_p + Kj * innov;
        Pj_up = (eye(n_x) - Kj*H) * Pj_p * (eye(n_x) - Kj*H)' + Kj*Rj*Kj';
        Pj_up = 0.5*(Pj_up + Pj_up');

        filt.models{j}.x = xj_up;
        filt.models{j}.P = Pj_up;

        innov_j(:,j) = innov;
        S_j(:,:,j)   = Sj;

        % ---- Gaussian likelihood: N(innov; 0, S) ----------------------------
        Sj_sym = 0.5*(Sj+Sj');
        [~, pd] = chol(Sj_sym);
        if pd == 0
            Lam(j) = exp(-0.5 * innov' / Sj_sym * innov) / ...
                     sqrt((2*pi)^n_z * det(Sj_sym) + eps);
        else
            Lam(j) = 1e-300;   % numerically singular: very low likelihood
        end
    end

    % ========================================================
    % STEP 3 & 4: Model probability update
    % ========================================================
    mu_new_unnorm = Lam .* c;
    mu_sum = sum(mu_new_unnorm);
    if mu_sum < eps
        mu_new = filt.mu;   % keep old weights if all likelihoods collapse
    else
        mu_new = mu_new_unnorm / mu_sum;
    end
    filt.mu = mu_new;

    % ========================================================
    % STEP 5: Fused estimate
    % ========================================================
    x_fuse = zeros(n_x,1);
    for j = 1:N
        x_fuse = x_fuse + mu_new(j) * filt.models{j}.x;
    end

    P_fuse = zeros(n_x,n_x);
    for j = 1:N
        delta = filt.models{j}.x - x_fuse;
        P_fuse = P_fuse + mu_new(j) * (filt.models{j}.P + delta*delta');
    end
    P_fuse = 0.5*(P_fuse + P_fuse');

    filt.x = x_fuse;
    filt.P = P_fuse;

    % ---- Outputs ------------------------------------------------------------
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
