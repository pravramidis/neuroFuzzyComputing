% Your main script
syms x y
f = x^2 + y^2 + (0.5*x + y)^2 + (0.5*x + y)^4;
% grad_f = [2 *x + (0.5* x + y) + 2*(0.5* x + y)^3; 
%                   2 *(0.5* x + 2 *y + 2 *(0.5 *x + y)^3)];
grad_f = [2 *x + 2*0.5*(0.5* x + y) + 4*0.5*(0.5* x + y)^3; 
                  2*y+2*(0.5* x + y) + 4 *(0.5 *x + y)^3];
disp(grad_f);
hessianMatrix = hessian(f,[x,y]);

% grad_f = @(x, y) [2*x + 0.5 + 2*(0.5*x + y)^3; 
%                   2*y + 2 + 4*(0.5*x + y)^3];

x0 = [3; 3];    % Initial guess
tol = 1e-6;     % Tolerance
max_iter = 5;   % Maximum number of iterations
   

% Initialize variables
z = x0;
% r = -grad_f(z(1), x(2));  % Residual and initial gradient
r_not = subs(-grad_f, [x,y], [z(1), z(2)]);
r = double(r_not);
d = r;          % Initial search direction
iter = 0;
% hessMatrix = hessian(f,[x,y]);
hess = subs(hessianMatrix, [x, y], [z(1), z(2)]);
hess_numeric = double(hess);

while norm(r) > tol && iter < max_iter
    iter = iter + 1;

    % Line search (could use more sophisticated method)
    % alpha = fminbnd(@(a) f(x(1) + a*d(1), x(2) + a*d(2)), 0, 1);
    alpha = (r' *r)/(r'* hess_numeric* r);

    % Print the current gradient, lambda (alpha), and x
    % disp(['Iteration ', num2str(iter)]);
    disp(['Iteration ', num2str(iter), ': Direction = ', mat2str(d'), ', Alpha = ', num2str(alpha), ', z_new = ', mat2str(z')]);

    % Update variables
    z_new = z + alpha * d;
    hess = subs(hessianMatrix, [x, y], [z_new(1), z_new(2)]);
    hess_numeric = double(hess);
    r_new_not = subs(-grad_f, [x,y], [z_new(1), z_new(2)]);
    r_new = double(r_new_not);
    beta = (r_new' * r_new) / (r' * r);
    d = r_new + beta * d;

    % disp([', z_new = ', mat2str(z_new')]);


    % Update for next iteration
    z = z_new;
    r = r_new;
end

