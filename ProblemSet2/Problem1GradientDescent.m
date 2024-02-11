% Define the function F(w)
F = @(w) w(1)^2 + w(2)^2 + (0.5*w(1) + w(2))^2 + (0.5*w(1) + w(2))^4;

% Define the gradient of the function F(w)
gradF = @(w) [2*w(1) + 2*(0.5*w(1) + w(2))*0.5 + 4*(0.5*w(1) + w(2))^3*0.5;
              2*w(2) + 2*(0.5*w(1) + w(2)) + 4*(0.5*w(1) + w(2))^3];

% Initialize the starting point w
w = [3; 3];

% Step size (learning rate)
alpha = 0.01;

% Perform 10 iterations of the Gradient Descent algorithm
for iter = 1:10
    % Update w using the gradient
    gradRes = gradF(w);
    w = w - alpha * gradF(w);
    
    % Display the current state
    fprintf('Iteration %d: gradres %f,%f\n', iter,gradRes(1), gradRes(2));
    fprintf('Iteration %d: w = [%f, %f], F(w) = %f\n', iter, w(1), w(2), F(w));
    
    % Stop if the change is smaller than the accuracy (three decimal points)
    if norm(gradF(w), 2) < 1e-3
        fprintf('Stopping criterion met\n');
        break;
    end
end
