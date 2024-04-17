% function parsave(fname, p_sample, LogLik, prior)
% save(fname, 'p_sample', 'LogLik', 'prior')
% end
function parsave(fname, p_em, par_sim, LogPost_em_all, LogPost_sim_all, LogPost_sim_acc, prior)
save(fname, 'p_em', 'par_sim', 'LogPost_em_all', 'LogPost_sim_all', 'LogPost_sim_acc', 'prior')
end