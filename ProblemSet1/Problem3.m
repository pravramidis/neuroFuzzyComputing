% Parameters
a = 0.16;  
b = 0.8; 

% Initial values
xstart = 0.5;
x0 = xstart; 
x1 = xstart;

% Number of iterations
num_iterations = 100; 

% Initialize arrays to store points
x_values = zeros(1, num_iterations + 1);
x_values(1) = x0;
x_values(2) = x1;

% Iterate to calculate x k+1
for k = 2:num_iterations
    x_values(k+1) = 1 - a*x_values(k)^2 + b*x_values(k-1);
end

% Plot the points
figure;
plot(0:num_iterations, x_values, 'o-');
xlabel('k');
ylabel('x_k');
title('Iterations of x_{k+1} = 1 - ax_k^2 + bx_{k-1}');

legend(['a = ', num2str(a), ', b = ', num2str(b), ', x_0 = ', num2str(x0)]);
grid on;
