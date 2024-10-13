function train_x = normal(train_x)
   [m,~]= size(train_x);  % 种群数 * 维度
   train_x = (train_x-repmat(min(train_x),m,1))./(repmat(max(train_x),m,1)-repmat(min(train_x),m,1)); 
   train_x =train_x';
end