imageArray = [
    20, 35, 35, 35, 35, 20;
    29, 46, 44, 42, 42, 27;
    16, 25, 21, 19, 19, 12;
    66, 120, 116, 154, 114, 62;
    74, 216, 174, 252, 172, 112;
    70, 210, 170, 250, 170, 110
    ];

test = [
    20, 35, 35, 35;
    29, 46, 44, 42;
    16, 25, 21, 19;
    28, 25, 19, 58;
    ];
% imshow(test, []);

kernel = [
    1, 1, 1;
    1, 0, 1;
    1, 1, 1
    ];

f1 = [
    -10, -10, -10;
    5, 5, 5;
    -10, -10, -10
    ];

f2 = [
    2 2 2;
    2 -12 2;
    2 2 2
    ];

f3 = [
    -20, -10, 0, 5, 10;
    -10, 0, 5, 10, 5;
    0, 5, 10, 5, 0;
    5, 10, 5, 0, -10;
    10, 5, 0, -10, -20
    ];


% imshow(imageArray, []);

convolvedImage = conv2(imageArray,kernel,"same");
convolvedF1 = conv2(imageArray,f1,"same");
convolvedF2 = conv2(imageArray,f2,"same");
convolvedF3 = conv2(imageArray,f3,"same");


% subplot(1, 2, 2);
% imshow(convolvedF1,[]);
% % imshow(convolvedF2,[]);

figure;

% Original image
subplot(3, 2, 1);
imshow(imageArray, []);
title('Original Image');

% Convolved image with filter f1
subplot(3, 2, 2);
imshow(convolvedF1, []);
title('Convolved Image with f1');

% Convolved image with filter f2
subplot(3, 2, 3);
imshow(imageArray, []);
title('Original Image');


% Original image
subplot(3, 2, 4);
imshow(convolvedF2, []);
title('Convolved Image with f2');


% Convolved image with filter f1
subplot(3, 2, 5);
imshow(imageArray, []);
title('Original Image');


% Convolved image with filter f3
subplot(3, 2, 6);
imshow(convolvedF3, []);
title('Convolved Image with f3');

