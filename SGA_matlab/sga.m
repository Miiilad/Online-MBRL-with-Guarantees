clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% systems definitions (see also system_.m)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inverted pendulum with our parameters
pendulum = system_;
pendulum.Q = eye(2);
pendulum.R = 2;
pendulum.x = sym('x',[2,1],'real'); 
%pendulum.dom = [-3*pi/4, 3*pi/4 ; -1, 1]; 
pendulum.dom = [-1, 1 ; -1, 1];
%pendulum.dom = [-3, 3 ; -10, 10]; 
pendulum.f = [pendulum.x(2); 19.6*sin(pendulum.x(1))- 4*pendulum.x(2)];
pendulum.g = [0; 40];
pendulum.u0 = -(1/40)*pendulum.x(1) - (19.6/40)*sin(pendulum.x(1));
%pendulum.u0 = -0.98*sin(pendulum.x(1));
%pendulum.u0 = -pendulum.x(1);

% cartpole
m = 0.1;
M = 1;
L = 0.8;
g = 9.8;
cartpole = system_;
cartpole.Q = diag([60, 1.5, 180, 45]);
cartpole.R = 1;
cartpole.x = sym('x',[4,1],'real');
x1 = cartpole.x(1);
x2 = cartpole.x(2);
x3 = cartpole.x(3);
x4 = cartpole.x(4);
c = 1/(M+m*sin(x1)*sin(x1));
%cartpole.dom = [-10, 10; -10, 10; -10, 10; -20, 20]; 
cartpole.dom = 0.5*[-0.1, 0.1; -0.1, 0.1; -0.1, 0.1; -0.1, 0.1]; 
cartpole.f = [x2; c*(-L*m*x2*x2*sin(x1)*cos(x1) + g*(M+m)*sin(x1))/L; x4; c*(m*sin(x1)*(L*x2*x2-g*cos(x1)))];
cartpole.g = [0; -c*cos(x1)/L; 0; c];
cartpole.u0 = -90.7474*x1 + -25.5800*x2 + -13.4164*x3 + -16.2467*x4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% select the system to use
sys = pendulum;

% second argument is the order of the polynomial with the highest order
% order must be even and greater than or equal to 2
basis = get_basis(sys.x, 2);

% number of iterations
iterations = 3;

% method with which to solve linear equation for coefficients
% choose 0 for exact, 1 for double, n > 1 for vpa with n digits
solve_method = 99; 

% check the residual at each step? choose 0 for speed and 1 for yes
check_residual = 0;

% print decimals instead of exact
sympref('FloatingPointOutput', false) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop (policy iteration)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

us = sym(zeros(iterations,1)); % vector for storing control iterates
Vs = sym(zeros(iterations,1)); % vector for storing value iterates
u = sys.u0; % initialize control
tic % start timer

for i = 1:iterations % do policy iteration
    iteration = i
    us(i) = u; % store the present u
    V = value(sys, u, basis, solve_method, check_residual); % get new value by solving GHJB with Galerkin
    Vs(i) = V; % store the present V
    u = control(sys, V); % get new control
    toc % record elapsed time
end

controller = us(end) % show iterates of control and value
valuee = Vs(end)

% find average cost on some domain
%domain = sys.dom; % use system domain, or use some other domain
%integral = Vs(end); % initialize integral
%area = 1; % initialize area
%for i = 1:size(sys.x,1) % integrate over the domain
%    integral = int(integral,sys.x(i),domain(i,1),domain(i,2));
%    area = area*(domain(i,2) - domain(i,1));
%end
%average_cost = integral/area % divide for average cost

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting solutions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiledlayout(1,2)
legend_string = string(1:iterations);
nexttile
hold on
for i = 1:iterations
    u_to_plot = us(i,1);
    if size(sys.x,1) > 1 % sub all but one variables to get a 2d plot
        for j = 2:size(sys.x,1)
            u_to_plot = subs(u_to_plot,sys.x(j),0);
        end
    end
    fplot(u_to_plot,sys.dom(1,:))
end
xlabel('position x1')
ylabel('control u')
title('Control vs position')
legend(legend_string)
nexttile
hold on
for i = 1:iterations
    V_to_plot = Vs(i,1);
    if size(sys.x,1) > 1 % sub all but one variables to get a 2d plot
        for j = 2:size(sys.x,1)
            V_to_plot = subs(V_to_plot,sys.x(j),0);
        end
    end
    fplot(V_to_plot,sys.dom(1,:))
end
xlabel('position x1')
ylabel('value V')
title('Value vs position')
legend(legend_string)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get value from control (p.33 in Beard)
function V = value(sys, u, basis, solve_method, check_residual)
    n_basis = size(basis,1); % number of basis vectors
    cs = sym('c',[n_basis,1],'real'); % make unknown coefficient vector
    V = dot(cs,basis); % generate unknown value
    error = ghjb(sys, V, u); % sub into ghjb to get error function
    disp('ghjb done')
    projs = sym(zeros(n_basis,1)); % vector for storing error projections
    for i = 1:n_basis % project error function onto basis
        vector = basis(i,1); % iterate thru basis vectors
        projs(i,1)=inner(sys.x,vector,error,sys.dom); % inner product
        disp('project')
    end
    [A,b] = equationsToMatrix(projs, cs); % get matrix form of resulting equation
    condition = cond(vpa(A)) % ill-conditioned for high dimensional bases
    if solve_method == 0
        cs_solved = A\b; % solve for cs symbolically
    elseif solve_method == 1
        cs_solved = double(A)\double(b); % solve for cs with float
    else
        cs_solved = vpa(A,solve_method)\vpa(b,solve_method); % use vpa
    end
    disp('solve done')
    V = dot(cs_solved,basis); % build value function
    if check_residual % check the residual if desired
        residual = max(abs(check_value(sys, u, V, basis))) % check the residual
    end
end

% check residuals of value
function res = check_value(sys, u, V, basis)
    n_basis = size(basis,1); % number of basis vectors
    error = ghjb(sys, V, u); % sub into ghjb to get error
    res = sym(zeros(n_basis,1)); % vector for storing residuals
    for i = 1:n_basis % project error function onto basis
        vector = basis(i,1); % iterate thru basis vectors
        res(i,1)=inner(sys.x,vector,error,sys.dom); % inner product
    end
end

% inner product (p.33 in Beard)
function product = inner(x,p,q,domain)
    product = p*q; %initialize the product
    for i = 1:size(x,1) % integrate over the domain
        lower = domain(i,1);
        upper = domain(i,2);
        product = int(product,x(i),lower,upper);
    end
end

% numerical inner product
function product = ninner(x,p,q,domain)
    product = p*q; %initialize the product
    
    % for pendulum
    %d = 0.1;
    %[a,s] = ndgrid(-4:d:4,-4:d:4);
    %arr = subs(product, {x(1),x(2)}, {a,s}); 
    %product = sum(arr,"all");

    % for cartpole
    d = 0.01;
    [a,s,m,l] = ndgrid(-0.05:d:0.05,-0.05:d:0.05,-0.05:d:0.05,-0.05:d:0.05);
      %[a,s,m,l] = ndgrid(-10:d:10,-10:d:10,-10:d:10,-20:d:20);
    arr = subs(vpa(product), {x(1),x(2),x(3),x(4)}, {a,s,m,l}) ;
    product = sum(arr,"all");
end

% get error generated by approximate V in GHJB (eq 3.7, p.25 in Beard)
function error = ghjb(sys, V, u) 
    error = gradient(V,sys.x).'*(sys.f+sys.g*u)+sys.q(sys.x)+sys.r(u);
end

% get control from value (p.26 in Beard)
function c = control(sys, V) 
    c = (-0.5)*inv(sys.R)*(sys.g).'*gradient(V,sys.x);
end

% get polynomial basis for galerkin (see p.75 in Beard for concrete example)
function basis = get_basis(x, order)
    dim = size(x,1); % dimension of the basis
    list = perms(dim,order); % all possible power combinations
    basis = sym([]); % initialize list of basis vectors
    for i = 1:size(list,1) % iterate over all possible power combos
        row = list(i,:); % get a particular combo and append if it is legal
        if (mod(sum(row), 2) == 0)&&(sum(row) <= order)&&(sum(row)>0)
            if ~ismember(prod(x.^(row.')),basis)
                basis(end+1) = prod(x.^(row.')); % append if not already
            end
        end
    end
    basis = basis.'; % change to column vector
end

% this method is used in get_basis
function vec = perms(dim, order)
    if dim ==1 % base case
         vec = (0:order).';
    else % do recursion to assemble all combinations
        vec = zeros((order+1)^dim,dim); % blank list of appropriate size
        for i = 0:(order) % iterate through list
            block = (order+1)^(dim-1);
            vec(i*block+1:(i+1)*block,1) = i*ones(block,1); % set left col
            vec(i*block+1:(i+1)*block,2:end) = perms(dim-1,order); % set other
        end
    end
end


% this method checks if every V is positive definite
% no need to check if v(0) = 0 since basis guarantees this to be true
% this uses symbolics, but does not work for higher dimensional bases
% because matlab cannot decide whether the condition is true
function pd = is_positive(Vs, sys)
    domain = sys.dom;
    x = sys.x;
    pd = 1;
    for i = 1:size(Vs,1)
        V = Vs(i);
        for j = 1:size(domain,1) % apply assumptions to each state variable
            assumeAlso(x(j) <= domain(i,2));  % consider only the domain
            assumeAlso(x(j) >= domain(i,1));
        end
        pd = pd*isAlways(V >= 0); 
    end
end