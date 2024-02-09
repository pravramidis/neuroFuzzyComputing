function output = problem15(image, kernel)
    [rows, cols] = size(image);
    k = size(kernel, 1);
    pad = floor(k / 2);
    paddedImage = padarray(image, [pad pad], 'replicate', 'both');
    output = zeros(rows, cols);
    
    for row = 1:rows
        for col = 1:cols
            region = paddedImage(row:row+k-1, col:col+k-1);
            output(row, col) = sum(sum(region .* kernel));
        end
    end
end

