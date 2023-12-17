% Patterns for class A
patternsA = [0, 0; 0, 1; 1, 0; -1, -1];

% Patterns for class B
patternsB = [2.1, 0; 0, -2.5; 1.6, -1.6];

syms x;
syms y;

% Scatter plot for class A
scatter(patternsA(:, 1), patternsA(:, 2), '*b', 'DisplayName', 'Class A');
hold on;

% Scatter plot for class B
scatter(patternsB(:, 1), patternsB(:, 2), '*r', 'DisplayName', 'Class B');

% Set axis labels and legend
xlabel('x');
ylabel('y');
legend('Location', 'best');
title('Plot of Patterns');
axis([-3 3 -3 3]);
x = linspace(-3, 3, 100);
y = x-1.5;
plot(x,y,'-g', 'DisplayName','Decision Boundary');

% Display grid
grid on;

% Display the plot
hold off;
