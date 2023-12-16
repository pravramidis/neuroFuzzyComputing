
xRange = [-20, 20];
yRange = [-20, 20];
p1 = [1 2]';
t1 = -1;
p2 = [-2 1]';
t2 = 1;
syms w1;
syms w2;
syms inp1;
syms inp2;
inp = [inp1 inp2];
probability = 0.5;


disp(p1);
disp(p2);

c = probability*t1^2+probability*t2^2;
fprintf("c = %d\n",c);
h = probability*t1*p1+probability*t2*p2;
% disp(h)
fprintf("h = %d\n",h);

R = probability*p1*(p1')+probability*p2*(p2');
disp(R);

f(w1,w2) = c-2*[w1 w2]*h+[w1 w2]*R*([w1 w2]');

xstar = inv(R)*h;
disp(xstar);
inp1 = linspace(-3, 3, 100);
inp2 = (-xstar(1)/xstar(2))*inp1;

point1 = xstar'*p1;
point2 = xstar'*p2;
disp(point1);
disp(point2);

figure;
hold on;
plot(inp1,inp2, 'LineWidth',2);
plot(p1(1),p1(2), '*');
plot(p2(1),p2(2), '*');
title('Decision Boundry');
% set(gca, 'XTickLabel', []);
% set(gca, 'YTickLabel', []);

hold off;
figure;
disp(f);

fcontour(f, [xRange, yRange]);
axis equal;
title('MSE Contour Plot');
xlabel('w1');
ylabel('w2');



% LMS Algorithm
alpha = 0.025;  % Learning rate
iterations = 100;  % Number of iterations

% Initial weights
W = [3; 1];

X = [1 2; -2 1];
y = [-1;1];  % Target values

W_history = zeros(iterations, length(W));

% LMS
for iter = 1:iterations

    predictions = X * W;

    errors = predictions - y;

    W = W - (alpha / size(X, 1)) * X' * errors;
    
    W_history(iter, :) = W;
end

% Generate a grid of W0 and W1 values
w0_vals = linspace(-4, 4, 1000);
w1_vals = linspace(-4, 4, 1000);
[W0, W1] = meshgrid(w0_vals, w1_vals);
    
% Compute the cost function values for each combination of W0 and W1
J_vals = zeros(size(W0));
for i = 1:numel(W0)
    J_vals(i) = 1/(2*size(X, 1)) * sum((X * [W0(i); W1(i)] - y).^2);
end
    
% Plot the contour plot
figure;
contour(W0, W1, J_vals, logspace(-2, 3, 20));
hold on;


% Plot the trajectory
plot(W_history(:, 1), W_history(:, 2), 'r*-', 'MarkerSize', 2, 'LineWidth', 1);
axis equal;
xlabel('W0');
ylabel('W1');
title('Trajectory of LMS Algorithm');
hold off;





