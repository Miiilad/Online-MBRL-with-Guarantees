function y = phi_fun(x)
% Basis function for V(x)
x1 = x(1);
x2 = x(2);

y  = [x1^2;
	  x2^2;
	  x1*x2;
	  x1^4;
	  x2^4;
	  x1^2*x2^2
	];

end