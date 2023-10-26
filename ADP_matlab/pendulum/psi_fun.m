function y = psi_fun(x)
x1 = x(1);
x2 = x(2);

y  = [x1 ;    
      x2  ;             
      x1*x1*x1   ;      
      x1*x1*x2  ;       
      x1*x2*x2  ;       
      x2*x2*x2          
	];
end