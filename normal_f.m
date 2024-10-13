function fake = normal_f(fake)
   [m,n]= size(fake);  % 种群数 * 维度
   fake = (fake-repmat(min(fake),m,1))./(repmat(max(fake),m,1)-repmat(min(fake),m,1)); 
   
end