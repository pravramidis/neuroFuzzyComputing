% Your main script
syms x y
f = x^2 + y^2 + (0.5*x + y)^2 + (0.5*x + y)^4;
grad_f = [2 *x + (0.5* x + y) + 2*(0.5* x + y)^3; 
                  2 *(0.5* x + 2 *y + 2 *(0.5 *x + y)^3)];
disp(grad_f);
hess = hessian(f,[x,y]);

% grad_f = @(x, y) [2*x + 0.5 + 2*(0.5*x + y)^3; 
%                   2*y + 2 + 4*(0.5*x + y)^3];

x0 = [3; 3];    % Initial guess
tol = 1e-6;     % Tolerance
max_iter = 5;   % Maximum number of iterations

[x, iter] = conjugateGradientFR(f, grad_f, x0, tol, max_iter, hess);




