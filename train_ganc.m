function [params_gen, params_dis,stGen,stDis] = train_ganc(Population,p_g,p_d,s_g,s_d,noise,maxepoch)
    train_x= Population.decs;
   
    train_y = Population.objs; 
    [m,n]= size(train_x);  % 种群数 * 维度
    train_x = normal(train_x);
    [~,n1]= size(train_y);  % 种群数 * 目标数
   
    %%  GAN的中间层 16 64 512 可以自己设置
    first = 16;
    second = 32;
    third = 128;

    %% Settings
    settings.latent_dim = n1;
    settings.batch_size = min(m,size(noise,1)); 
    settings.lrD = 0.0001; settings.lrG = 0.0001; settings.beta1 = 0.5;
    settings.beta2 = 0.999; settings.maxepochs =maxepoch;
    %% Initialization
    if size(p_g,1) == 0 || size(p_d,1) == 0
        [paramsGen,stGen] = initialize_G();  
        [paramsDis,stDis]  =initialize_D();  
    else
        paramsGen = p_g;
        paramsDis = p_d;
        stGen = s_g;
        stDis = s_d;
    end
 
    avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
    %% Train
    numIterations = floor((m + settings.batch_size -1)/settings.batch_size);
    out = false; epoch = 1; global_iter = 0;
    while ~out
        tic; 
        trainXshuffle = train_x(:,randperm(size(train_x,2)));
        if mod(epoch,10) == 0
            fprintf('Epoch %d\n',epoch) 
        end
        for i=1:numIterations
            global_iter = global_iter+1;
             noise1 =  noise(randperm(settings.batch_size),:);
             noise1 = dlarray(noise1.','CB');
            idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
            if(size(idx,2) > size(trainXshuffle,2))
                idx = idx(1,1:size(trainXshuffle,2));
            end
          
            XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');
            % XBatch=dlarray(single(trainXshuffle(:,size(idx,2))),'CB');
            
            [~,GradDis,stGen,stDis] = ...
                    dlfeval(@modelGradients,XBatch,noise1,...
                    paramsGen,paramsDis,stGen,stDis);
    %%
            % Update Discriminator network parameters
            [paramsDis,avgG.Dis,avgGS.Dis] = ...
                adamupdate(paramsDis, GradDis, ...
                avgG.Dis, avgGS.Dis, global_iter, ...
                settings.lrD, settings.beta1, settings.beta2);
            [GradGen,~,stGen,stDis] = ...
                    dlfeval(@modelGradients,XBatch,noise1,...
                    paramsGen,paramsDis,stGen,stDis);    
            % Update Generator network parameters
            [paramsGen,avgG.Gen,avgGS.Gen] = ...
                adamupdate(paramsGen, GradGen, ...
                avgG.Gen, avgGS.Gen, global_iter, ...
                settings.lrG, settings.beta1, settings.beta2);   
        end
    
       
        epoch = epoch+1;
        if epoch == settings.maxepochs 
            params_gen = paramsGen;
            params_dis = paramsDis;
            stGen = stGen;
            out = true;
        end    
    end
    
%%   初始化generatoe
    function [paramsGen,stGen] = initialize_G()
        paramsGen.FCW1 = dlarray(...
        initializeGaussian([first,settings.latent_dim],.02));     % 16
        paramsGen.FCb1 = dlarray(zeros(first,1,'single'));    
        paramsGen.BNo1 = dlarray(zeros(first,1,'single'));
        paramsGen.BNs1 = dlarray(ones(first,1,'single'));
        paramsGen.FCW2 = dlarray(initializeGaussian([second,first]));
        paramsGen.FCb2 = dlarray(zeros(second,1,'single'));           % 128
        paramsGen.BNo2 = dlarray(zeros(second,1,'single'));
        paramsGen.BNs2 = dlarray(ones(second,1,'single'));
        paramsGen.FCW3 = dlarray(initializeGaussian([third,second]));    % 1024
        paramsGen.FCb3 = dlarray(zeros(third,1,'single'));
        paramsGen.BNo3 = dlarray(zeros(third,1,'single'));
        paramsGen.BNs3 = dlarray(ones(third,1,'single'));
        paramsGen.FCW4 = dlarray(initializeGaussian(...
            [n,third]));
        paramsGen.FCb4 = dlarray(zeros(n,1,'single'));
        stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];
    end
%%   初始化D
    function [paramsDis,stDis] = initialize_D()
        paramsDis.FCW1 = dlarray(initializeGaussian([third,n],.02));
        paramsDis.FCb1 = dlarray(zeros(third,1,'single'));
        paramsDis.BNo1 = dlarray(zeros(third,1,'single'));
        paramsDis.BNs1 = dlarray(ones(third,1,'single'));
        paramsDis.FCW2 = dlarray(initializeGaussian([second,third]));
        paramsDis.FCb2 = dlarray(zeros(second,1,'single'));
        paramsDis.BNo2 = dlarray(zeros(second,1,'single'));
        paramsDis.BNs2 = dlarray(ones(second,1,'single'));
        paramsDis.FCW3 = dlarray(initializeGaussian([first,second]));    % 16
        paramsDis.FCb3 = dlarray(zeros(first,1,'single'));
        paramsDis.FCW4 = dlarray(initializeGaussian([1,first]));
        paramsDis.FCb4 = dlarray(zeros(1,1,'single'));
        stDis.BN1 = []; stDis.BN2 = [];
    end
    %% Weight initialization
    function parameter = initializeGaussian(parameterSize,sigma)
    if nargin < 2
        sigma = 0.05;
    end
    parameter = randn(parameterSize, 'single') .* sigma;
    end
    
    %% Generator
    function [dly,st] = Generator(dlx,params,st)
  
    dly = fullyconnect(dlx,params.FCW1,params.FCb1);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);

    dly = fullyconnect(dly,params.FCW2,params.FCb2);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);

    dly = fullyconnect(dly,params.FCW3,params.FCb3);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);

    dly = fullyconnect(dly,params.FCW4,params.FCb4);
    % tanh
    dly = tanh(dly);
   
    end
    %% Discriminator
    function [dly,st] = Discriminator(dlx,params,st)
  
    dly = fullyconnect(dlx,params.FCW1,params.FCb1);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);
   
    dly = fullyconnect(dly,params.FCW2,params.FCb2);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);

    dly = fullyconnect(dly,params.FCW3,params.FCb3);
    dly = leakyrelu(dly,0.2);
    dly = dropout(dly);
   
    dly = fullyconnect(dly,params.FCW4,params.FCb4);
   
    dly = sigmoid(dly);
    end
    %% gpu dl array wrapper
    function dlx = gpdl(x,labels)
    dlx = dlarray(x,labels);
    end
    %% modelGradients
    function [GradGen,GradDis,stGen,stDis]=modelGradients(x,z,paramsGen,...
        paramsDis,stGen,stDis)                    %(@modelGradients,XBatch,noise,...
    [fake_images,stGen] = Generator(z,paramsGen,stGen);
    d_output_real = Discriminator(x,paramsDis,stDis);
    
    fake_images = normal_f(fake_images);
  
    [d_output_fake,stDis] = Discriminator(fake_images,paramsDis,stDis);
    
    % Loss due to true or not
    d_loss = -mean(.9*log(d_output_real+eps)+log(1-d_output_fake+eps));
    g_loss = -mean(log(d_output_fake+eps));
    
    % For each network, calculate the gradients with respect to the loss.
    GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
    GradDis = dlgradient(d_loss,paramsDis);
    end
    %% dropout
    function dly = dropout(dlx,p)
    if nargin < 2
        p = .3;
    end
    n_d = p*10;
    mask = randi([1,10],size(dlx));
    mask(mask<=n_d)=0;
    mask(mask>n_d)=1;
    dly = dlx.*mask;
    
    end
end