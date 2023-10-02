function Xi = s_parm_psi(Yv, sigma)

Xi = (norm(Yv)-1)/2 + sqrt(((1+norm(Yv)^2)/4 - sigma));

end
