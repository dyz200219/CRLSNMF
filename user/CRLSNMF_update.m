function [V_star, obj, error_cnt] = CRLSNMF_update(X, label, varargin)
% Input: X(m*n)

pnames = {'maxiter', 'tolfun', 'beta', 'gamma', 'sigma'};
dflts  = {150, 1e-6, 0, 0};
[maxiter, tolfun, beta, gamma, sigma] = internal.stats.parseArgs(pnames,dflts,varargin{:});

view_num = length(X); 
class_num = length(unique(label)); 

U = cell(view_num, 1); 
V = cell(view_num, 1); 
S = cell(view_num, 1); % S
Y = cell(view_num, 1); % Y = X - UV'
obj = zeros(1, 1); 

for i = 1: view_num
    [X{i}, ~] = data_normalization(X{i}, [], 'std');
end

% construct graph
A = cell(view_num, 1); %  similarity matrix
D = cell(view_num, 1); % diagonal degree matrix
AT = cell(view_num, 1); 
options = [];
options.WeightMode = 'Binary';
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5; % Number of neighbors

for view_idx = 1: view_num
    A{view_idx} = constructA(X{view_idx}',options);
    D{view_idx} = diag(sum(A{view_idx}, 1));
    AT{view_idx} = D{view_idx}\ A{view_idx};
end

for view_idx = 1: view_num
    [U{view_idx}, V{view_idx}] = KMeansdata(X{view_idx}, class_num); 
    U{view_idx} = abs(U{view_idx});
    V{view_idx} = abs(V{view_idx});
    Y{view_idx} = X{view_idx} - U{view_idx}*V{view_idx};
    
    tao = zeros(length(Y{view_idx}), 1);
    
    for i = 1:length(Y{view_idx})
        tao(i) = sigma;
    end
    
    S{view_idx} = constructS(Y{view_idx}, tao);
end



% 更新错误计数器
error_cnt = 0;

% 迭代更新
for iter = 1: maxiter
    
   % update V^p(p = 1, ..., P)
   for view_idx = 1: view_num
       
       V{view_idx} = V{view_idx} .* ((U{view_idx}')*(X{view_idx}-S{view_idx})+beta*calc_sum_V(V, view_idx)+(gamma/view_num)*V{view_idx}*(AT{view_idx}')) ./...
           max((U{view_idx}')*U{view_idx}*V{view_idx}+beta*(view_num-1)*V{view_idx}+(gamma/(2*view_num))*(V{view_idx}*AT{view_idx}*(AT{view_idx}')+V{view_idx}), eps); 
       
   end
   
   % update U^p(p = 1, ..., P)
   for view_idx = 1: view_num
         
       U{view_idx} = U{view_idx} .* ((X{view_idx}-S{view_idx})*(V{view_idx}')) ./...
              max(U{view_idx}*V{view_idx}*(V{view_idx}'), eps);
  
   end
   
  
    
   % objective function value
   obj(iter) = calc_obj_value(X, U, V, S, AT, beta, gamma);
   fprintf('iter = %d, obj = %g\n', iter, obj(iter));
   
   % If the value of the objective function increases after the iteration, the error counter is increased by 1
   if (iter>=2)&&(obj(iter)>obj(iter-1))
      error_cnt = error_cnt + 1; 
   end
   
   % algorithmic stopping condition
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<tolfun)|| iter==maxiter
        break;
    end
   
end

V_star = zeros(size(V{view_num})); % V*
for view_idx = 1: view_num
    V_star = V_star + V{view_idx};
end
V_star = V_star/view_num; 

end

function [obj_value] = calc_obj_value(X, U, V, S, AT, beta, gamma)
view_num = length(X); 
obj_value = 0;
for view_idx = 1: view_num
    obj_value = obj_value...
        + (norm((X{view_idx}-S{view_idx})-U{view_idx}*V{view_idx}, 'fro').^2)...
        + beta*calc_sum_d(V, view_idx)...
        + (gamma/(2*view_num))*trace(norm(AT{view_idx}*V{view_idx}'-V{view_idx}','fro').^2);
end
end

function [sum_V] = calc_sum_V(V, view_idx)
view_num = length(V); 
sum_V = zeros(size(V{1}));
for i = 1: view_num
   if i ~=  view_idx
       sum_V = sum_V + V{i};
   end
end
end

function [sum_V_d] = calc_sum_d(V, view_idx)
view_num = length(V);
sum_V_d = 0;
for i = 1: view_num
    if i ~=  view_idx
        sum_V_d = sum_V_d + norm(V{view_idx}-V{i},'fro').^2;
    end
end
end

