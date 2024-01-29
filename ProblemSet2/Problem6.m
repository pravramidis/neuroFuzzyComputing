
p1 = 0;
p2 = 1;
t = 0.75;
w1 = 1;
w2 = -1;
w1_2 = 0.5;
b1 = 1;
alpha = 1;

%define maxiter
maxiter = 1;


% Define the tanh activation function and its derivative
tanh_activation = @(x) tanh(x);
tanh_derivative = @(x) sech(x).^2;

for iter = 1:maxiter
    % Calculate the input to the neuron (n)
    n = w1 * p1 + w2 * p2 + w1_2 * (p1 * p2) + b1;
    
    % Calculate the output of the neuron (a)
    a = tanh_activation(n);
    
    % Calculate the error (E)
    E = 0.5 * (t - a)^2;

    
    
    % Calculate the gradients of the error with respect to each weight
    % and the bias term using the chain rule
    dE_da = -(t - a);
    da_dn = tanh_derivative(n);
    
    % Since p1 is 0, dn/dw1 is also 0, hence grad_w1 is 0
    grad_w1 = dE_da * da_dn * p1;
    grad_w1_2 = dE_da * da_dn * (p1 * p2);
    
    % Calculate other gradients
    grad_w2 = dE_da * da_dn * p2;
    grad_b1 = dE_da * da_dn;
    
    % Update the weights and bias
    w1_new = w1 - alpha * grad_w1;
    w2_new = w2 - alpha * grad_w2;
    w1_2_new = w1_2 - alpha * grad_w1_2;
    b1_new = b1 - alpha * grad_b1;

    w1 = w1_new;
    w2 = w2_new;
    w1_2 = w1_2_new;
    b1 = b1_new;
    
    % Display the new weights and bias
    disp(['Iter: ', num2str(iter)]);
    disp(['New w1: ', num2str(w1_new)]);
    disp(['New w2: ', num2str(w2_new)]);
    disp(['New w1_2: ', num2str(w1_2_new)]);
    disp(['New b1: ', num2str(b1_new)]);
    disp(['Error E: ', num2str(E)]);
    disp(['Output: ', num2str(a)]);

    if abs(a-t) < 1e-6
         break;
    end

end


