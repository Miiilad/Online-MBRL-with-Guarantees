function sim_trajectories(w, t_range, trajectories, dom)
    function dx = sys_wrapper(t,x) % no augmented state but must pass args
                u = dot(w,psi_fun(x)); % control to apply
                x1 = x(1);
                x2 = x(2);
                x3 = x(3);
                x4 = x(4);
                m = 0.1; % constants
                M = 1;
                L = 0.8;
                g = 9.8;
                % dimension of dx must match dim(x)!
                c = 1/(M+m*sin(x1)*sin(x1));
                dx = [x2; c*(-L*m*x2*x2*sin(x1)*cos(x1) + g*(M+m)*sin(x1))/L;...
                    x4; c*(m*sin(x1)*(L*x2*x2-g*cos(x1)))] + [0; -c*cos(x1)/L; 0; c]*u;
    end
range = dom(1,2)-dom(1,1); % this for random initial conditions
mid = 0.5*(dom(1,2)+dom(1,1));
figure % plot trajectories
xlabel('time t')
ylabel('position x1')
title(strcat("sample trajectories under control ", num2str(w(1:4).'), " ..."))
hold on
    for i = 1:trajectories
        x0 = range*(rand(4,1)-0.5*ones(4,1)) + mid*ones(4,1);
        [t,x] = ode45(@sys_wrapper, t_range, x0);
        plot(t,x(:,1))
    end
end
