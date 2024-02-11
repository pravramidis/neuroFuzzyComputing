function Problem5
    
    S1_values = [12];
    
    num_epochs = 10000; 
    dropout_probs = [0.15, 0.25, 0.35];

    alpha = 0.1;

    p = linspace(-2, 2, 100)';
    t = 1 + sin(p * 3 * pi / 8);

  
    for S1 = S1_values
        for prob = dropout_probs
            [W1, b1, W2, b2] = train_network(p, t, S1, alpha, num_epochs, prob);
            
            y = simulate_network(p, W1, b1, W2, b2);
            plot_results(p, t, y, S1, prob);
        end
    end
end

function [W1, b1, W2, b2] = train_network(p, t, S1, alpha, num_epochs, dropout_prob)
    
    W1 = rand(S1, 1) - 0.5;
    b1 = rand(S1, 1) - 0.5;
    W2 = rand(1, S1) - 0.5;
    b2 = rand() - 0.5;

   
    for epoch = 1:num_epochs
        for i = 1:length(p)
           
            dropout_mask = rand(S1, 1) > dropout_prob;

           
            n1 = W1 * p(i) + b1;
            a1 = logsig(n1);
            a1 = a1 .* dropout_mask; 
            n2 = W2 * a1 + b2;
            a2 = max(0, n2); 

           
            e = t(i) - a2;
            s2 = e; 
            s1 = (W2' * s2) .* (a1 .* (1 - a1)) .* dropout_mask; 

            
            W2 = W2 + alpha * s2 * a1';
            b2 = b2 + alpha * s2;
            W1 = W1 + alpha * s1 * p(i)';
            b1 = b1 + alpha * s1;
        end
    end
end


function y = simulate_network(p, W1, b1, W2, b2)
    
    a1 = logsig(W1 * p' + b1);
    y = max(0, W2 * a1 + b2);
end

function plot_results(p, t, y, S1, prob)
    
    figure;
    plot(p, t, 'b-', p, y, 'r--');
    title(sprintf('Hidden Neurons: %d, dropout probability: %.2f', S1, prob));
    xlabel('Input p');
    ylabel('Target and Network Output');
    legend('Target', 'Network Output');
end


