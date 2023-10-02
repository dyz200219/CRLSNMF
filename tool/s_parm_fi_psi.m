function fi_psi = s_parm_fi_psi (psi, Yv, sigma)

fi_psi = ((psi - norm(Yv))^2)/2 + sigma*log(1+psi);

end