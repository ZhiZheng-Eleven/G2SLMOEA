function guide_point = generate_ganpointc(Problem,noise,paramsGen,stGen)

  
        noise = dlarray(noise.','CB');   %randn([n1,100])
        result = Generator(noise,paramsGen,stGen);
       
        result = result.reshape([],Problem.D);
        
        I1 = rescale(result(:,1:Problem.M-1),Problem.lower(1,1),Problem.upper(1,Problem.M-1));
        I2 = rescale(result(:,Problem.M:end),Problem.lower(1,Problem.M),Problem.upper(1,Problem.M));
        result = [I1,I2];
        
        guide_point = single(gatext(result));

end
%% Generator
function [dly,stGen] = Generator(dlx,params,stGen)
    dly = fullyconnect(dlx,params.FCW1,params.FCb1);
    dly = leakyrelu(dly,0.2);

    dly = fullyconnect(dly,params.FCW2,params.FCb2);
    dly = leakyrelu(dly,0.2);

    dly = fullyconnect(dly,params.FCW3,params.FCb3);
    dly = leakyrelu(dly,0.2);
    dly = fullyconnect(dly,params.FCW4,params.FCb4);
    % tanh
    dly = tanh(dly);
end
function x = gatext(x)
x = extractdata(x);
end