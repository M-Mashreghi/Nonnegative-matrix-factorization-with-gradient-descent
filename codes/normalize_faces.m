function V_normalized = normalize_faces(V)
    V = dlmread('face.txt', ' ');
    
    V_normalized = zeros(size(V));
    
    for i = 1:size(V, 2)
        face = V(:, i);
        
        median_face = median(face);
        face = face - median_face + 0.5;
        
        median_deviation = median(abs(face - 0.5));
        face = 0.5 + (face - 0.5) * (0.25 / median_deviation);
        
        face(face < 1e-4) = 1e-4;
        face(face > 1) = 1;
        
        V_normalized(:, i) = face;
    end
    
    dlmwrite('face_normalized.txt', V_normalized, ' ');
end
