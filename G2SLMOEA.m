classdef G2SLMOEA < ALGORITHM
%<large> <multi/many> <real>
% Large-Scale multiobjective optimization Driven by
% Generative Adversarial Networks
% FE_GAN --- 0.1 --- The stage of guided by GAN
% alg --- 4 --- Select a alg 1 = CLIA, 2=NSGAII, 3=IBEA, 4 = G2S



%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    methods
        function main(Algorithm,Problem)
            [FE_GAN,alg] = Algorithm.ParameterSet(0.1,4);        
            Population = Problem.Initialization();
            [Z,~] = UniformPoint(Problem.N,Problem.M);    
            [uniform_points,~] = UniformPoint(Problem.N *(Problem.M-1),Problem.M);
            if alg==2
                [~,FrontNo,CrowdDis] = EnvironmentalSelection_NSGA_II(Population,Problem.N);
            end
            %% 初始化GAN        
            epoch = 5;    %根据自己需求设置，论文中有对epoch的敏感性分析
            stGen = [];
            stDis = [];  
            params_gen = [];
            params_dis = [];
            F1 = Population((NDSort(Population.objs,1)==1));     
            ARV = Problem.N;    
            flag = true;
            %% 执行算法
            while Algorithm.NotTerminated(Population)            
                times = (sum(mean(F1.objs,1),2)); 
                if(times > 1000 && ARV < Problem.N/3 && flag)                        
                    flag = false;
                end
                %第一阶段
                if(Problem.FE/Problem.maxFE < FE_GAN && flag)                                   
                    noise =  times  * uniform_points;  
                    [params_gen, ~,stGen,stDis] = train_ganc(Population,params_gen,params_dis,stGen,stDis,noise,epoch);
                    guide_point = generate_ganpointc(Problem,noise,params_gen,stGen);
                    [Offspring,guide_point] = generate_guidepointc(guide_point,Population,Problem);            
                    [Population, F1,ARV] = cascade_clusterc([Population, Offspring,guide_point], Z, 'PDM', Problem.N, Problem.FE < Problem.maxFE);  
                    
                %第二阶段，可以选择不同策略的MOEAs
                else
                    if alg==1
                        MatingPool = TournamentSelection(2, Problem.N, sum(max(0, Population.cons), 2)); 
                        Offspring  = OperatorGA(Population(MatingPool));
                        [Population, F1] = cascade_clusterc([Population, Offspring], Z, 'PDM', Problem.N, Problem.FE < Problem.maxFE);
                    end
                    %%
                    if alg == 2
                        MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                        Offspring  = OperatorGA(Population(MatingPool));
                        [Population,FrontNo,CrowdDis] = EnvironmentalSelection_NSGA_II([Population,Offspring],Problem.N);
                    end
                    %%
                    if alg==3
                        MatingPool = TournamentSelection(2,Problem.N,-CalFitness_IBEA(Population.objs,0.05));
                        Offspring  = OperatorGA(Population(MatingPool));
                        Population = EnvironmentalSelection_IBEA([Population,Offspring],Problem.N,0.05);
                    end                         
                    %%
                    if alg == 4
                        Offspring = generate_offspring(F1,Population,Problem,ARV);
                        [Population, F1,ARV] = cascade_clusterc([Population, Offspring], Z, 'PDM', Problem.N, Problem.FE < Problem.maxFE);  
                    end 
                end
            end  
        end
    end
end
