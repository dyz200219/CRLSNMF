function S = constructS (Yv, sigma)

i_num = size(Yv, 1);
j_num = size(Yv, 2);

psi = zeros(i_num, 1);
fi_psi = zeros(i_num, 1);
S = zeros(i_num, j_num);

for i_idx = 1 : i_num
    
    psi(i_idx) = s_parm_psi(Yv(i_idx, :), sigma(i_idx));
    fi_psi(i_idx) = s_parm_fi_psi(psi(i_idx), Yv(i_idx, :), sigma(i_idx));

    if fi_psi(i_idx) <= (norm(Yv(i_idx, :))^2)/2 && (1+norm(Yv(i_idx, :))^2 > 4*sigma(i_idx)) && psi(i_idx) > 0
        S(i_idx, :) = ((psi(i_idx)/norm(Yv(i_idx, :))) * Yv(i_idx, :));
    end
    
end
