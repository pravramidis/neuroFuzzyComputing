% Define the objective function
syms w1 w2
F = w1^2 + w2^2 + (0.5*w1 + w2)^2 + (0.5*w1 + w2)^4;

% Set initial point
x0 = [1; 1];

% Set convergence criterion
epsilon = 1e-7;

% Set maximum number of iterations
max_iter = 100;

% Initialize variables
xk = x0;

% Optimization loop
for k = 1:max_iter
    % STEP-1: Calculate gradient and Hessian
    grad_fk = double(subs(gradient(F, [w1; w2]), [w1; w2], xk));
    hess_fk = double(subs(jacobian(gradient(F, [w1; w2]), [w1; w2]), [w1; w2], xk));

    % STEP-2: Compute minimizing direction
    sk = -inv(hess_fk) * grad_fk;

   
    lambda_k = 1;
    

    % STEP-4: Find the next point
    xk_1 = xk + lambda_k * sk;

    % STEP-5: Check convergence criterion
    if norm(grad_fk) < epsilon
        disp(['Converged at iteration ', num2str(k)]);
        break;
    end

    % Update xk for the next iteration
    xk = xk_1;
end

disp('Optimal solution:');
disp(xk_1);
% Display the function value and the current solution
disp(['Function Value at xk: ', num2str(double(subs(F, [w1; w2], xk)))]);


