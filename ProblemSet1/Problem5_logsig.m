
function [a1,a2,a] = Problem5_logsig(p);


w111 = -2;
w121 = -1;
w211 = 2;
w212 = 1;

b1 = -0.5;
b2 = -0.75;
b = 0.5;

%logsig
n1 = w111*p+b1;
n2 = w121*p+b2;

a1 = sigmoid(n1);
a2 = sigmoid(n2);

%purelin
n = a1*w211 + a2*w212 + b;

a = linear(n);

end



function a = sigmoid(n)
    a = 1/(1+exp(-n));
end

function a = linear(n)
    a = n;
end


