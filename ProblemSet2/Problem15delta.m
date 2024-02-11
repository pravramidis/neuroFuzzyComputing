function output = Problem15delta(image, kernel, delta)
    [rows, cols] = size(image);
    k = size(kernel, 1);
    pad = floor(k / 2);
    paddedImage = padarray(image, [pad pad], 'replicate', 'both');
    output = zeros(rows, cols);
    
    for row = 1:rows
        col = 1;
        while col <= cols
            deltaCols = min(delta, cols - col + 1);
            for d = 0:deltaCols-1
                region = paddedImage(row:row+k-1, col+d:col+d+k-1);
                output(row, col+d) = sum(sum(region .* kernel));
            end
            col = col + delta;
        end
    end
end
