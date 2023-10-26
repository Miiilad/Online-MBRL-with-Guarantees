function [phi_diff,psi_e, psi_psi, q_x, x_end] = sys_cartpole(t_range, x0, w, a, Q)
    dim_x = size(x0,1);
    dim_psi = size(psi_fun(x0),1);
    [t,z] = ode45(@sys_wrapper, t_range, [x0; zeros(dim_psi + dim_psi*dim_psi + 1,1)]);
    function dz = sys_wrapper(t,z) % do this because we need augmented state z
        x = z(1:dim_x); % the state of the system
        e = sum(a*sin([1 3 7 11 13]*t)); % sinusoidal exploration noise
        u = dot(w(1:4),x); % control to apply
        x1 = x(1);
        x2 = x(2);
        x3 = x(3);
        x4 = x(4);
        m = 0.1; % constants
        M = 1;
        L = 0.8;
        g = 9.8;
        c = 1/(M+m*sin(x1)*sin(x1));
        dx = [x2; c*(-L*m*x2*x2*sin(x1)*cos(x1) + g*(M+m)*sin(x1))/L; x4;...
            c*(m*sin(x1)*(L*x2*x2-g*cos(x1)))] + [0; -c*cos(x1)/L; 0; c]*(e+u);
        psipsi = psi_fun(x)*(psi_fun(x).'); 
        psie = psi_fun(x)*e;
        q = (x.')*Q*x; % state cost
        dz = [dx; psie; reshape(psipsi,[dim_psi*dim_psi,1]); q];
    end
    % package stuff into augmented state z
    phi_diff = phi_fun(z(end,1:dim_x).') - phi_fun(z(1,1:dim_x).');
    psi_e = z(end, dim_x+1: dim_x + dim_psi).'; 
    psi_psi = z(end,dim_x + dim_psi + 1: dim_x + dim_psi + dim_psi*dim_psi).';
    psi_psi = reshape(psi_psi,[dim_psi,dim_psi]);
    q_x = z(end,end);
    x_end = z(end, 1:dim_x).';
end

% here is how z is allocated: 
% parts of each row of z:
% [     x,       psi_e,        psi_psi,       q_x]
% corresponding lengths:
% [dim(x),    dim(psi),     dim(psi)^2,         1]