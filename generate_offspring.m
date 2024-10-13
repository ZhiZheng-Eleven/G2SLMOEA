function Off = generate_offspring(GuidingSolution,P,Problem,ARV)   %CF是级联聚类得到的，C是GAN生成的
    times = (sum(mean(P.objs,1),2)); 
    p_obj = P.objs;
    g_obj = GuidingSolution.objs;
    tg = sum(mean(g_obj,1),2);
    distance_P = sqrt(sum(p_obj.^2,2));
    distance_g = sqrt(sum(g_obj.^2,2));  % 100 * 1 
    np = size(p_obj,1);
    np1 = size(g_obj,1);
    p_obj = (p_obj-repmat(min(p_obj),np,1))./(repmat(max(p_obj),np,1)-repmat(min(p_obj),np,1));
    g_obj = (g_obj-repmat(min(g_obj),np1,1))./(repmat(max(g_obj),np1,1)-repmat(min(g_obj),np1,1));
    Cosine =  1 - pdist2(single(p_obj),single(g_obj),'cosine');
    
    if times > 1
        [~,index] = max(Cosine,[],2); 
    else
        [~,index] = min(Cosine,[],2); 
    end
   
    if ARV <= 4 && tg < 5
        [~,index] = min(Cosine,[],2);
    end
     pro = class(Problem);
     
     if strcmp(pro,'MLSMOP2')
         [~,index] = min(Cosine,[],2); 
         
     end
      if strcmp(pro,'MLSMOP4')
         [~,index] = min(Cosine,[],2); 
         
      end
    
    index = index.';   % 1* 100
   
    p_dec = P.decs;
    g_dec = GuidingSolution.decs;
    dis = zeros(size(P,2),1);
   
   
    %% 
    
    for i = 1 : size(P,2)
        dis(i) = sqrt(sum((p_dec(i,:) - g_dec(index(1,i),:)).^2,2));
       
    end
    new_solution = zeros(Problem.N,Problem.D);
  
     for i = 1 : size(P,2)
        if distance_P(i,:) <= distance_g(index(1,i),:)
            direction = (g_dec(index(1,i),:) - p_dec(i,:))/dis(i);  %/dis(i)
           
        else 
            direction = (p_dec(i,:) - g_dec(index(1,i),:))/dis(i);  
           
        end
        direction = (p_dec(i,:) - g_dec(index(1,i),:))/dis(i);
        new_solution(i,:) =  p_dec(i,:)+ rand .* direction ; % rand .* direction
       
     end

    P1 =p_dec;
    
    for i = 1 : size(new_solution,1)
        index1 = randperm(size(P1 ,1),2);  
        
        new_solution(i,:) = new_solution(i,:) +    rand *(P1(index1(1,1),:) - P1(index1(1,2),:));        
    end
       PopX = max(min(repmat(Problem.upper,size(new_solution,1),1),new_solution),repmat(Problem.lower,size(new_solution,1),1));
    Off = pol_mutation(single(PopX),Problem);    
end

%% Polynomial mutation
function Offspring = pol_mutation(Offspring,Problem)
    N = size(Offspring,1);
    D = Problem.D;
    disM = 20;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site  = rand(N,D) < 1/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = SOLUTION(Offspring);
end
