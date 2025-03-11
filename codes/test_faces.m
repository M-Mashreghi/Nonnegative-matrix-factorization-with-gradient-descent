function [objective_val_history, H] = test_faces(W, testV_normalized)
    max_iter = 300;
    
    [m, n_test] = size(testV_normalized);
    k = size(W, 2);
    
    H = rand(k, n_test);
    
    objective_val_history = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        H = H .* ((W' * testV_normalized) ./ (W' * W * H ));
        objective_val_history(iter) = norm(testV_normalized - W * H, 'fro')^2;
    end

    
    figure('Name','Test Reconstruction Convergence','Color','w');
    plot(1:max_iter, objective_val_history, 'LineWidth',2, 'Color','r');
    xlabel('Iteration Number', 'FontSize',12);
    ylabel('Objective Value (||V - WH||_F^2)', 'FontSize',12);
    title('Convergence of Test Reconstruction', 'FontSize',14);
    grid on;
    
end
