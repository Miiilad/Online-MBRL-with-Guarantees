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
ball_lb = 0.1

xlim = [3.0,10.0]

P = np.load('P_verify.npy')
P_nonzero = P [1:,1:] # delete the first row and column

W = np.load('W_Wj.npy')

phi = [1,x1,x2,sin(x1),sin(x2)]

# expression of V
V = Expression(0)
for i in range(5):
    V_element = Expression(0)
    for j in range(5):
        V_element += P[j,i]*phi[j]
    V += V_element*phi[i]

print("V = ", V)

##########################################
# calculate optimal control u

# calculate Wphi of the optimal control
w_optimal = W[:,5:10]

m,n = w_optimal.shape
wphi = []

for i in range(m):
  w_phi = Expression(0)
  for j in range(n):
    w_phi += phi[j]*w_optimal[i,j]
  wphi.append(w_phi)

# expression of u

first_half = []

for i in range(n):
  first_element = Expression(0)
  for j in range(n):
    first_element += phi[j]*P[i,j]
  first_half.append(first_element)

# partial * second half
second_half = [0]
second_half.append(wphi[0])
second_half.append(wphi[1])
second_half.append(wphi[0]*cos(x1))
second_half.append(wphi[1]*cos(x2))

# optimal control u
u = Expression(0)
for i in range(len(first_half)):
  u += first_half[i]*second_half[i]

R = 1
u = -u/R

##########################################

print("u = ", u)

f = [x2, -4.*x2 + 19.6*sin(x1) + 40.*u]

x1_bound = logical_and(x[0]<=xlim[0], x[0]>=-xlim[0])
x2_bound = logical_and(x[1]<=xlim[1], x[1]>=-xlim[1])
x12_bound = logical_and(x1_bound, x2_bound)
ball= Expression(0)

for i in range(len(x)):
    ball += x[i]*x[i]
ball_in_bound = logical_and(ball_lb*ball_lb <= ball,x12_bound)

lie_derivative_of_V = Expression(0)
for i in range(len(x)):
	lie_derivative_of_V += f[i]*V.Differentiate(x[i]) 

reach = logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon) # condition on derivative of V

condition = reach 

start_ = timeit.default_timer() 

result = CheckSatisfiability(logical_not(condition),config)

stop_ = timeit.default_timer() 

t = stop_ - start_

print(f"Verification results for SOL.")

if (result): 
  print(f"Not a Lyapunov function. Found counterexample: ")
  print(result)
else:
  print("Satisfy conditions with eps = ", epsilon)
  print(V, f" is a Lyapunov function.")

print("Verification time =", t)
