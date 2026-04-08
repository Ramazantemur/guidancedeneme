function plot_results(log_circular, log_guidance, params)
% PLOT_RESULTS  Visualise the full simulation: circular phase + BPNG phase.
%
%   log_circular : struct with fields t, x,y,z, Vx,Vy,Vz, accel
%   log_guidance : struct with fields t, x,y,z, Vx,Vy,Vz, accel,
%                                     los_rate, range, in_fov
%   params       : simulation params struct (for target_pos etc.)

    % ============================================================
    % Figure 1: 3D Trajectory
    % ============================================================
    figure('Name','UAV Trajectory','NumberTitle','off','Color','w');
    hold on; grid on; axis equal;

    % Circular phase
    plot3(log_circular.y, log_circular.x, -log_circular.z, ...
          'b-', 'LineWidth', 2, 'DisplayName','Circular orbit');

    % BPNG phase
    plot3(log_guidance.y, log_guidance.x, -log_guidance.z, ...
          'r-', 'LineWidth', 2, 'DisplayName','BPNG guidance');

    % Target
    tp = params.target_pos;
    plot3(tp(2), tp(1), -tp(3), 'kp', ...
          'MarkerSize', 14, 'MarkerFaceColor','k', 'DisplayName','Target');

    % Start marker
    plot3(log_circular.y(1), log_circular.x(1), -log_circular.z(1), ...
          'go', 'MarkerSize', 10, 'MarkerFaceColor','g', 'DisplayName','Start');

    xlabel('East (m)'); ylabel('North (m)'); zlabel('Altitude (m)');
    title('UAV 3D Trajectory: Circular Orbit → BPNG Terminal Guidance');
    legend('Location','best');
    view(30, 30);

    % ============================================================
    % Figure 2: Time histories
    % ============================================================
    t_all    = [log_circular.t;   log_circular.t(end) + log_guidance.t];
    alt_circ = -log_circular.z;
    alt_guid = -log_guidance.z;
    alt_all  = [alt_circ; alt_guid];

    a_mag_circ = vecnorm(log_circular.accel, 2, 2);
    a_mag_guid = vecnorm(log_guidance.accel, 2, 2);
    a_all      = [a_mag_circ; a_mag_guid];

    figure('Name','Time Histories','NumberTitle','off','Color','w');

    % --- Altitude ---
    subplot(3,1,1);
    t_phase2_start = log_circular.t(end);
    plot(log_circular.t, alt_circ, 'b-', 'LineWidth',1.5); hold on;
    plot(log_circular.t(end) + log_guidance.t, alt_guid, 'r-', 'LineWidth',1.5);
    xline(t_phase2_start, '--k', 'BPNG Start', 'LabelHorizontalAlignment','left');
    xlabel('Time (s)'); ylabel('Altitude (m)');
    title('Altitude vs Time');
    legend('Circular','BPNG','Location','best');
    grid on;

    % --- Acceleration magnitude ---
    subplot(3,1,2);
    g = params.g;
    plot(log_circular.t, a_mag_circ/g, 'b-', 'LineWidth',1.5); hold on;
    plot(log_circular.t(end) + log_guidance.t, a_mag_guid/g, 'r-', 'LineWidth',1.5);
    xline(t_phase2_start, '--k');
    xlabel('Time (s)'); ylabel('|a| (g)');
    title('Total Acceleration Command');
    legend('Circular','BPNG','Location','best');
    grid on;

    % --- Range to target ---
    subplot(3,1,3);
    range_circ = sqrt((log_circular.x - tp(1)).^2 + ...
                      (log_circular.y - tp(2)).^2 + ...
                      (log_circular.z - tp(3)).^2);
    plot(log_circular.t, range_circ, 'b-', 'LineWidth',1.5); hold on;
    plot(log_circular.t(end) + log_guidance.t, log_guidance.range, 'r-', 'LineWidth',1.5);
    xline(t_phase2_start, '--k');
    xlabel('Time (s)'); ylabel('Range (m)');
    title('Range to Target');
    legend('Circular','BPNG','Location','best');
    grid on;

    % ============================================================
    % Figure 3: LOS rates (guidance phase only)
    % ============================================================
    figure('Name','LOS Rates (BPNG Phase)','NumberTitle','off','Color','w');
    subplot(3,1,1);
    plot(log_guidance.t, rad2deg(log_guidance.los_rate(:,1)), 'r-','LineWidth',1.5);
    ylabel('\omega_x (deg/s)'); grid on; title('LOS Angular Rate (NED)');
    subplot(3,1,2);
    plot(log_guidance.t, rad2deg(log_guidance.los_rate(:,2)), 'g-','LineWidth',1.5);
    ylabel('\omega_y (deg/s)'); grid on;
    subplot(3,1,3);
    plot(log_guidance.t, rad2deg(log_guidance.los_rate(:,3)), 'b-','LineWidth',1.5);
    ylabel('\omega_z (deg/s)'); xlabel('Time (s)'); grid on;

    fprintf('\n=== SIMULATION COMPLETE ===\n');
    final_pos  = [log_guidance.x(end), log_guidance.y(end), log_guidance.z(end)];
    miss_dist  = norm(final_pos - tp');
    fprintf('  Miss distance : %.2f m\n', miss_dist);
    fprintf('  BPNG duration : %.1f s\n', log_guidance.t(end));
    fprintf('===========================\n\n');
end
