% Generate a 2000x2000 random image
image = randi([0, 255], 228, 228);

% Define kernels
kernels = {
    struct('size', 3, 'kernel', ones(3, 3) / 9);
    struct('size', 7, 'kernel', ones(7, 7) / 49);
    struct('size', 11, 'kernel', ones(11, 11) / 121);
};

% Initialize variables for average CPU times
averageCpuTimeHorizontal = zeros(length(kernels), 1);
averageCpuTimeDelta = zeros(length(kernels), 7);

% Number of iterations for averaging
numIterations = 20;

% Loop over each kernel
for i = 1:length(kernels)
    kernel = kernels{i}.kernel;
    fprintf('Kernel size: %dx%d\n', kernels{i}.size, kernels{i}.size);
    
    % Measure execution time for delta = 1 for 5 iterations
    cpuTimes = zeros(numIterations, 1);
    for iter = 1:numIterations
        startCpuTime = cputime;
        output = problem15(image, kernel); % Your convolution function
        cpuTimes(iter) = cputime - startCpuTime;
    end
    averageCpuTimeHorizontal(i) = mean(cpuTimes);
    fprintf('Average Horizontal Scan CPU Time (delta=1): %.4f seconds\n', averageCpuTimeHorizontal(i));
    
    % Measure execution time for delta = 1 to 10
    for delta = 1:20
        cpuTimes = zeros(numIterations, 1);
        for iter = 1:numIterations
            startCpuTime = cputime;
            output = Problem15b(image, kernel, delta); % Your modified convolution function
            cpuTimes(iter) = cputime - startCpuTime;
        end
        averageCpuTimeDelta(i, delta) = mean(cpuTimes);
        fprintf('Average Delta Wide Strip CPU Time (delta=%d): %.4f seconds\n', delta, averageCpuTimeDelta(i, delta));
    end
end
