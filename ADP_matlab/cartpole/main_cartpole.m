%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CARTPOLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters for ADP simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
n_phi = size(phi_fun(ones(1,4)),1); % number of unknowns c
n_psi = size(psi_fun(ones(1,4)),1); % number of unknowns w
intervals = max((n_phi+n_psi),160); % collect at least as many as the number of unknowns

Q = diag([60, 1.5, 180, 45]); % state cost 
R = 1; % control cost

%w0 = [90.7474;25.5800;13.4164;16.2467;zeros(n_psi-4,1)]; % initial control used for ADP simulation
%w0 = [510.2;138.9;283.4;143.7;zeros(n_psi-4,1)]; % initial control used for ADP simulation
w0 = [49;12;8;8;zeros(n_psi-4,1)]; % initial control used for ADP simulation

% other system parameters found in sys_cartpole.m

dt = 0.02; % length of integration interval
a = 0.2; % amplitude of noise
k = 0.2; % controls IC, domain of plotting, and ICs of random trajectories
x0 = k*[1;1;1;1]; % intial condition for ADP simulation

iterations = 10; % number of off-policy iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop for ADP simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% arrays for data from ADP
phi_diffs = zeros(n_phi,intervals);
psi_es = zeros(n_psi,intervals);
psi_psis = zeros(n_psi,n_psi,intervals);
q_xs = zeros(1,intervals);

tic % start timer
for i = 1:intervals
    %interval = i % print interval number
    [phi_diff,psi_e, psi_psi, q_x, x_end] = sys_cartpole(dt*[i-1,i], x0, w0, a, Q); % integrate
    phi_diffs(:,i) = phi_diff; % save data
    psi_es(:,i) = psi_e;
    psi_psis(:,:,i) = psi_psi;
    q_xs(:,i) = q_x;
    x0 = x_end; % update x0
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% off-policy iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% assemble unknown c and w into column vector [c;w] and solve for it
ws = zeros(n_psi,iterations); % vector for storing control coeff iterates
cs = zeros(n_phi,iterations); % vector for storing value coeff iterates
ws(:,1) = w0; % save initial control coeffs
wi = w0; % initialize control
Ai = zeros(n_phi+n_psi, n_phi+n_psi); % intervals matrix
bi = zeros(n_phi+n_psi, 1); % forcing term
for i = 1:iterations 
    % get new system and solve for u,v
    for j = 1:intervals
        f = phi_diffs(:,j); % temporary vars for ease of typing-in
        e = psi_es(:,j);
        p = psi_psis(:,:,j);
        q = q_xs(:,j);
        Ai(j,:) = [f.', (2*R*(e+p*(w0-wi))).']; % obtained these by hand
        bi(j,:) = -R*(wi.')*p*wi-q;
    end
    cw = Ai\bi; % solve system
    c = cw(1:n_phi); % separate c and w
    w = cw(n_phi+1:end);
    ws(:,i+1) = w; % store the present control
    cs(:,i) = c; % store the present value
    wi = w; % update control iterate for the next system
end
toc % record elapsed time
ws(:,end) % show final control

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dom = k*[-1,1]; % domain for horizonal axis plotting
us = sym(zeros(iterations,1)); % vector for storing control iterates
Vs = sym(zeros(iterations,1)); % vector for storing value iterates
x = sym('x',[4,1],'real'); % symbolic state
psi_sym = psi_fun(x); % symbolic bases
phi_sym = phi_fun(x);

for i = 1:iterations
    us(i,1) = dot(psi_sym, ws(:,i));
    Vs(i,1) = dot(phi_sym, cs(:,i));
end

tiledlayout(1,2)
legend_string = string(1:iterations);
nexttile
hold on
for i = 1:iterations
    u_to_plot = us(i,1);
    if size(x,1) > 1 % sub all but one variables to get a 2d plot
        for j = 2:size(x,1)
            u_to_plot = subs(u_to_plot,x(j),0);
        end
    end
    fplot(u_to_plot, dom)
end
xlabel('position x1')
ylabel('control u')
title('Control vs position')
legend(legend_string)
nexttile
hold on
for i = 1:iterations
    V_to_plot = Vs(i,1);
    if size(x,1) > 1 % sub all but one variables to get a 2d plot
        for j = 2:size(x,1)
            V_to_plot = subs(V_to_plot,x(j),0);
        end
    end
    fplot(V_to_plot, dom)
end
xlabel('position x1')
ylabel('value V')
title('Value vs position')
legend(legend_string)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulating random trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
us(end)
Vs(end)
rng(1); % seed
dom = k*[-1,1];
t_range = [0,5];
trajectories = 20; %  get this many random ICs
sim_trajectories(ws(:,1),t_range, trajectories, dom) % simulate with initial control
sim_trajectories(ws(:,end),t_range, trajectories, dom) % simulate with final control
