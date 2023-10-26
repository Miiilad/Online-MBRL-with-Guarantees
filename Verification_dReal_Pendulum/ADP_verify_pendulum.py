import numpy as np
from dreal import *
import timeit 

x1 = Variable("x1")
x2 = Variable("x2")
x = [x1,x2]

config = Config()
config.use_polytope_in_forall = True
config.use_local_optimization = True
config.precision = 1e-2
epsilon = 0.
ball_lb = 0.3
xlim = [1.5,2.1] # set upper bounds for x

# expression of V
V = -(8348705711791185*x1**4)/72057594037927936 - (8055519091113769*x1**2*x2**2)/1152921504606846976 + (6394020526263095*x1**2)/4503599627370496 + (4895481769477553*x1*x2)/36028797018963968 + (6438708792788647*x2**4)/295147905179352825856 + (2371300731736623*x2**2)/72057594037927936

##########################################
# optimal control u
u = (7289152652914787*x1**3)/36028797018963968 - (7418046106097609*x1**2*x2)/288230376151711744 + (4129724914470083*x1*x2**2)/576460752303423488 - (3049887435953181*x1)/2251799813685248 - (8713938879005657*x2**3)/18446744073709551616 - (5951073488495957*x2)/9007199254740992

f = [x2, -4.*x2 + 19.6*sin(x1) + 40.*u] # dynamics

x1_bound = logical_and(x[0]<=xlim[0], x[0]>=-xlim[0])
x2_bound = logical_and(x[1]<=xlim[1], x[1]>=-xlim[1])
x12_bound = logical_and(x1_bound, x2_bound)

ball= Expression(0)

for i in range(len(x)):
    ball += x[i]*x[i]
ball_in_bound = logical_and(ball_lb*ball_lb <= ball, x12_bound)

lie_derivative_of_V = Expression(0)
for i in range(len(x)):
	lie_derivative_of_V += f[i]*V.Differentiate(x[i]) 

reach = logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon) # condition on derivative of V

condition = reach 

start_ = timeit.default_timer() 

result = CheckSatisfiability(logical_not(condition),config)

stop_ = timeit.default_timer() 

t = stop_ - start_

print(f"Verification results for ADP.")

if (result): 
  print(f"Not a Lyapunov function. Found counterexample: ")
  print(result)
else:
  print("Satisfy conditions with eps = ", epsilon)
  print(V, f" is a Lyapunov function.")

print("Verification time =", t)
