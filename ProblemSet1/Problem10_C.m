% ADALINE with LMS Training Algorithm

% Input patterns for class A
patterns_A = [0 0; 0 1; 1 0; -1 -1];

% Input patterns for class B
patterns_B = [2.1 0; 0 -2.5; 1.6 -1.6];

% Combine input patterns
patterns = [patterns_A; patterns_B];

% Target outputs (class A = 1, class B = -1)
targets = [ones(size(patterns_A, 1), 1); -ones(size(patterns_B, 1), 1)];
disp(targets);

% Add bias term to patterns
%patterns = [patterns, ones(size(patterns, 1), 1)];

% Initial weights and biases
weights = 0.5 * ones(2, 1);
bias = 0.5;
disp(weights);

% Learning rate
learning_rate = 0.01;

% Maximum number of iterations
max_iterations = 10000;

% LMS training algorithm
for iteration = 1:max_iterations
    % Calculate predicted output
    output = patterns * weights + bias;
    
    % Calculate error
    error = targets - output;
    disp(error);
    
    % Update weights and bias
    weights = weights + 2*learning_rate * (patterns' * error);
    bias = bias + 2*learning_rate * sum(error);
    
    % Check for convergence (stop if the error is small)
    if max(abs(error)) < 1e-6
        break;
    end
end

% Display final weights and bias
fprintf('Final Weights: %s\n', mat2str(weights));
fprintf('Final Bias: %f\n', bias);

% Plot decision boundary
figure;
scatter(patterns_A(:, 1), patterns_A(:, 2), 'bo', 'Marker', '*');
hold on;
scatter(patterns_B(:, 1), patterns_B(:, 2), 'rx', 'Marker', '*');
xlabel('x');
ylabel('y');
title('ADALINE with LMS Training');
legend('Class A', 'Class B');

% Plot decision boundary
x_decision = min(patterns(:, 1)):0.1:max(patterns(:, 1));
y_decision = (-weights(1) * x_decision - bias) / weights(2);
plot(x_decision, y_decision, 'g--', 'LineWidth', 2);
axis([-2 3 -3 2]);
legend('Class A', 'Class B', 'Decision Boundary');
hold off;


