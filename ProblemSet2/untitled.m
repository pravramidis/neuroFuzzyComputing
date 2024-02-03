syms x y
f =  x^2 + y^2 + (0.5*x + y)^2 + (0.5*x + y)^4;
hess = hessian(f,[x,y]);
hess_2x2 = [hess(1, 1), hess(1, 2); hess(2, 1), hess(2, 2)];
disp(hess_2x2);

x_val = 3;
y_val = 3;

% Substitute the values into the Hessian matrix
hess_at_values = subs(hess, [x, y], [x_val, y_val]);
hess_numeric = double(hess_at_values);
disp(hess_numeric);

A = [x+y x; y y];
disp(A);