
    % Objective function
    F = @(w) 0.1*w(1)^2 + 2*w(2)^2;
    %F = @(w) 0.1*(w(1)+w(2))^2 + 2*(w(1) - w(2))^2;


    % Gradient of the objective function
    gradF = @(w) [0.2*w(1); 4*w(2)];

    % Initial guess
    w = [1; 1]; % Adjust as needed

    % Adadelta parameters
    alpha = 3;
    rho = 0.95;
    epsilon = 1e-6;
    delta_w = zeros(size(w));
    E_g2 = zeros(size(w));
    E_delta_w2 = zeros(size(w));

    % For plotting
    w1_range = -2:0.01:2;
    w2_range = -2:0.01:2;
    [W1, W2] = meshgrid(w1_range, w2_range);
    F_val = 0.1*W1.^2 + 2*W2.^2;


    figure; contour(W1, W2, F_val, 50); hold on;

    % Optimization loop
    for iter = 1:100000
        % Compute gradient
        g = gradF(w);

        % Update E_g2
        E_g2 = rho * E_g2 + (1 - rho) * g.^2;

        % Compute update
        delta_w = -sqrt((E_delta_w2 + epsilon)./(E_g2 + epsilon)) .* g;

        % Update E_delta_w2
        E_delta_w2 = rho * E_delta_w2 + (1 - rho) * delta_w.^2;

        % Update weights
        w = w + alpha * delta_w;

        % Plotting the trajectory
        plot(w(1), w(2), 'r.', 'MarkerSize', 10);

        % Check convergence (optional)
        if norm(g) < 1e-3
            disp(iter);
            break;
        end
    end
    hold off;

