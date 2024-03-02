rng(42);
g = @(p) 1 + sin(p*pi/8);

num_points = 30;


%p_train = -4 + 8*rand(num_points, 1); %genarates random points
p_train = linspace(-4, 4, num_points)'; %generates uniform points


num_centers = [4, 8, 12, 20];
learning_rate = 0.1;

for num_centers_idx = 1:length(num_centers)
    
    centers = -4 + 8*rand(num_centers(num_centers_idx), 1);
    
    
    w = randn(num_centers(num_centers_idx), 1);
    b = randn;
    
   
    max_iterations = 1000;
    error_threshold = 1e-6;
    prev_error = Inf;
    iteration = 0;
    
    while true
        iteration = iteration + 1;
        
        h = zeros(num_points, num_centers(num_centers_idx));
        for i = 1:num_points
            for j = 1:num_centers(num_centers_idx)
                h(i, j) = exp(-norm(p_train(i) - centers(j))^2);
            end
        end
        y = h * w + b;
        
        error = sum((y - g(p_train)).^2) / num_points;
        
        delta_w = -2 * (y - g(p_train))' * h / num_points;
        delta_b = -2 * sum(y - g(p_train)) / num_points;
        w = w + learning_rate * delta_w';
        b = b + learning_rate * delta_b;
        
        if abs(prev_error - error) < error_threshold || iteration >= max_iterations
            break;
        end
        
        prev_error = error;
    end
    
    p_test = linspace(-4, 4, 100);
    h_test = zeros(length(p_test), num_centers(num_centers_idx));
    for i = 1:length(p_test)
        for j = 1:num_centers(num_centers_idx)
            h_test(i, j) = exp(-norm(p_test(i) - centers(j))^2);
        end
    end
    y_test = h_test * w + b;
    
    figure;
    plot(p_test, y_test, 'r', 'LineWidth', 2);
    hold on;
    scatter(p_train, g(p_train), 'bo', 'LineWidth', 1.5);
    xlabel('p');
    ylabel('g(p)');
    title(['Approximation with ', num2str(num_centers(num_centers_idx)), ' centers']);
    legend('Approximation', 'Training points');
    
    SSE = sum((y - g(p_train)).^2);
    disp(['Sum Squared Error with ', num2str(num_centers(num_centers_idx)), ' centers: ', num2str(SSE)]);
end
