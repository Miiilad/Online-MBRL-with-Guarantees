function y = psi_fun(x)
% basis functions for control
% this includes partial derivatives of every phi wrt every state variable
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

y = [     
      x1;
      x2;
      x3;
      x4;
    x4^3;
 x3*x4^2;
 x3^2*x4;
    x3^3;
 x2*x4^2;
x2*x3*x4;
 x2*x3^2;
 x2^2*x4;
 x2^2*x3;
    x2^3;
 x1*x4^2;
x1*x3*x4;
 x1*x3^2;
x1*x2*x4;
x1*x2*x3;
 x1*x2^2;
 x1^2*x4;
 x1^2*x3;
 x1^2*x2;
    x1^3];
end