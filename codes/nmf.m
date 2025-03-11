function [W, H, obj_vals] = nmf(V, k, max_iter)

    [m, n] = size(V);
    
    W = rand(m, k);
    H = rand(k, n);
    
    obj_vals = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        H = H .* ((W' * V) ./ (W' * W * H ));
        
        W = W .* ((V * H') ./ (W * H * H' ));
        
        obj_vals(iter) = norm(V - W * H, 'fro')^2;
    end
end
