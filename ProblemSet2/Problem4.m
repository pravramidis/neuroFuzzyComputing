function Problem4
    % Experiment parameters
    S1_values = [2, 8, 12]; % Different sizes of the hidden layer
    learning_rates = [0.01, 0.1, 0.2]; % Different learning rates
    num_epochs = 10000; % Number of training epochs

    % Generate training data
    p = linspace(-5, 5, 100)';
    t = 1 + sin(p * 3 * pi / 8);

    % Experiment with different configurations
    for S1 = S1_values
        for alpha = learning_rates
            [W1, b1, W2, b2] = train_network(p, t, S1, alpha, num_epochs);
            % Evaluate the trained network
            y = simulate_network(p, W1, b1, W2, b2);
            plot_results(p, t, y, S1, alpha);
        end
    end
end

function [W1, b1, W2, b2] = train_network(p, t, S1, alpha, num_epochs)
    % Initialize weights and biases
    W1 = rand(S1, 1) - 0.5;
    b1 = rand(S1, 1) - 0.5;
    W2 = rand(1, S1) - 0.5;
    b2 = rand() - 0.5;

    % Training loop
    for epoch = 1:num_epochs
        for i = 1:length(p)
            % Forward pass
            n1 = W1 * p(i) + b1;
            a1 = logsig(n1);
            n2 = W2 * a1 + b2;
            a2 = max(0, n2); % ReLU activation

            % Backpropagation
            e = t(i) - a2;
            s2 = e; % ReLU derivative
            s1 = W2' * s2 .* a1 .* (1 - a1); % logsig derivative

            % Update weights and biases
            W2 = W2 + alpha * s2 * a1';
            b2 = b2 + alpha * s2;
            W1 = W1 + alpha * s1 * p(i)';
            b1 = b1 + alpha * s1;
        end
    end
end

function y = simulate_network(p, W1, b1, W2, b2)
    % Simulate the network for given inputs
    a1 = logsig(W1 * p' + b1);
    y = max(0, W2 * a1 + b2);
end

function plot_results(p, t, y, S1, alpha)
    % Plot the results
    figure;
    plot(p, t, 'b-', p, y, 'r--');
    title(sprintf('Hidden Neurons: %d, Learning Rate: %.2f', S1, alpha));
    xlabel('Input p');
    ylabel('Target and Network Output');
    legend('Target', 'Network Output');
end


