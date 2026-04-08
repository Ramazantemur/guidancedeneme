function [los_vec, los_rate, filt] = seeker_alpha(los_noisy, filt, dt, params)
% SEEKER_ALPHA  Sliding-window (moving average) filter for uLOS.
%
%   This is Algorithm 1 — a pure signal-processing approach.
%   The filter maintains a FIFO buffer of the last W LOS unit-vectors
%   and averages them (mean then renormalise) to suppress noise.
%   LOS rate is computed as the finite difference of successive averages.
%
%   State stored in filt:
%     filt.buf   : (3 x W) buffer of recent unit-LOS vectors (NaN = empty)
%     filt.ptr   : circular buffer write pointer (1-based)
%     filt.count : number of valid entries in buffer
%     filt.prev  : previous smoothed LOS vector (for rate estimate)
%
%   params fields:
%     .alpha_window : number of samples in sliding window (default 10)
%
%   Outputs:
%     los_vec  : (3x1) smoothed LOS unit vector
%     los_rate : (3x1) LOS angular-rate estimate (rad/s)
%     filt     : updated filter state

    W = params.alpha_window;   % window length

    % ---- initialise on first call -----------------------------------------------
    if isempty(filt.buf)
        filt.buf   = nan(3, W);
        filt.ptr   = 1;
        filt.count = 0;
        filt.prev  = los_noisy;
    end

    % ---- insert new measurement into circular buffer ----------------------------
    filt.buf(:, filt.ptr) = los_noisy;
    filt.ptr   = mod(filt.ptr, W) + 1;
    filt.count = min(filt.count + 1, W);

    % ---- compute window average -------------------------------------------------
    valid_cols = filt.buf(:, ~any(isnan(filt.buf), 1));
    if isempty(valid_cols)
        los_avg = los_noisy;
    else
        los_avg = mean(valid_cols, 2);
    end
    % renormalise to unit vector
    n = norm(los_avg);
    if n < 1e-6
        los_avg = los_noisy;
    else
        los_avg = los_avg / n;
    end

    % ---- LOS rate by finite difference of smoothed vector ----------------------
    los_rate = (los_avg - filt.prev) / dt;

    % ---- store & output ---------------------------------------------------------
    filt.prev = los_avg;
    los_vec   = los_avg;
end
