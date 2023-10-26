classdef system_
   properties
       x, f, g, Q, R, u0, dom
   end
   methods
       function state_cost = q(obj,x)
           state_cost = x.' * obj.Q * x;
       end

       function control_cost = r(obj,u)
            control_cost = u.' * obj.R * u;
       end
 
   end
end