

% Define the range of p values
p_values = linspace(-2, 2, 100); % You can adjust the number of points as needed

% Initialize arrays to store the results
a1_values = zeros(size(p_values));
a2_values = zeros(size(p_values));
a_values = zeros(size(p_values));

% Calculate a1, a2, and a for each p value
for i = 1:length(p_values)
    [a1, a2, a] = Problem5_logsig(p_values(i));
    a1_values(i) = a1;
    a2_values(i) = a2;
    a_values(i) = a;
end

% Plot a1
figure;
plot(p_values, a1_values, '-*');
title('Plot of a1 (logsig)');
xlabel('p');
ylabel('a1');

% Plot a2
figure;
plot(p_values, a2_values, '-*');
title('Plot of a2 (logsig)');
xlabel('p');
ylabel('a2');

% Plot a
figure;
plot(p_values, a_values, '-*');
title('Plot of a (logsig)');
xlabel('p');
ylabel('a');

% Combine all plots into a single figure
figure;
hold on;
title('Plot of a,a1,a2 (logsig)');
plot(p_values, a1_values, '-*r');
plot(p_values, a2_values, '-*b');
plot(p_values, a_values, '-*m');
legend('a1', 'a2', 'a');
xlabel('p');
ylabel('a');
hold off;

% Calculate a1, a2, and a for each p value
for i = 1:length(p_values)
    [a1, a2, a] = Problem5_Swish(p_values(i));
    a1_values(i) = a1;
    a2_values(i) = a2;
    a_values(i) = a;
end

% Plot a1
figure;
plot(p_values, a1_values, '-*');
title('Plot of a1 (Swish)');
xlabel('p');
ylabel('a1');

% Plot a2
figure;
plot(p_values, a2_values, '-*');
title('Plot of a2 (Swish)');
xlabel('p');
ylabel('a2');

% Plot a
figure;
plot(p_values, a_values, '-*');
title('Plot of a (Swish)');
xlabel('p');
ylabel('a');

% Combine all plots into a single figure
figure;
hold on;
title('Plot of a,a1,a2 (Swish)');
plot(p_values, a1_values, '-*r');
plot(p_values, a2_values, '-*b');
plot(p_values, a_values, '-*m');
legend('a1', 'a2', 'a');
xlabel('p');
ylabel('a');
hold off;

