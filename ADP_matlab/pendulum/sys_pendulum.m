function [phi_diff,psi_e, psi_psi, q_x, x_end] = sys_pendulum(t_range, x0, w, a, Q)
    dim_x = size(x0,1);
    dim_psi = size(psi_fun(x0),1);
    [t,z] = ode45(@sys_wrapper, t_range, [x0; zeros(dim_psi + dim_psi*dim_psi + 1,1)]);

    
    function dz = sys_wrapper(t,z)
        e = sum(a*sin([1 3 7 11 13 15]*t)); % sinusoidal exploration noise
        x = z(1:dim_x); % the state of the system
        u = dot(w(1:2),x);
        % dimension of dx must match dim(x)!
        % u0 in this case is -x1
        dx = [x(2); 19.6*sin(x(1))- 4*x(2) + 40*(u+e)];  

        psipsi = psi_fun(x)*(psi_fun(x).');
        psie = psi_fun(x)*e;
        q = (x.')*Q*x; % state cost
        dz = [dx; psie; reshape(psipsi,[dim_psi*dim_psi,1]); q];
    end

    phi_diff = phi_fun(z(end,1:dim_x).') - phi_fun(z(1,1:dim_x).');
    psi_e = z(end, dim_x+1: dim_x + dim_psi).'; 
    psi_psi = z(end,dim_x + dim_psi + 1: dim_x + dim_psi + dim_psi*dim_psi).';
    psi_psi = reshape(psi_psi,[dim_psi,dim_psi]);
    q_x = z(end,end);
    x_end = z(end, 1:dim_x).';


end

% here is how z is allocated: 
% parts of each row of z with corresponding lengths:
% [     x,       psi_e,        psi_psi,       q_x]
% [dim(x),    dim(psi),     dim(psi)^2,         1]