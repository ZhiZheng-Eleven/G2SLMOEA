function [flag_GAN,flag_guide,count_NARV] = judge(Problem,ARV,times,count_NARV,flag_GAN,flag_guide)
    if(ARV<Problem.N/25)
        count_NARV = count_NARV + 1;
        if times < 10
            flag_guide = "MIN";
        end
        if(count_NARV>20)
            flag_GAN = false;
        end
    else
        flag_guide = "MAX";
    end                
end

