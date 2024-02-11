
syms w1 w2
F = w1^2 + w2^2 + (0.5*w1 + w2)^2 + (0.5*w1 + w2)^4;

x0 = [3; 3];

epsilon = 1e-7;

max_iter = 100;

xk = x0;

for k = 1:max_iter
    
    grad_fk = double(subs(gradient(F, [w1; w2]), [w1; w2], xk));
    hess_fk = double(subs(jacobian(gradient(F, [w1; w2]), [w1; w2]), [w1; w2], xk));

    sk = -inv(hess_fk) * grad_fk;
   
    lambda_k = 1;
    
    xk_1 = xk + lambda_k * sk;

    if norm(grad_fk) < epsilon
        disp(['Converged at iteration ', num2str(k)]);
        break;
    end

    xk = xk_1;
end
disp('Optimal solution:');
disp(xk_1);

disp(['Function Value at xk: ', num2str(double(subs(F, [w1; w2], xk)))]);


