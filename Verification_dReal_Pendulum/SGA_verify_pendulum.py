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

xlim = [3.0,9.2] # set bounds for x

# expression of V
V = - 0.000045075921492717859465867435151832*x1**8 - 0.000013820624916825717899355419486694*x1**7*x2 - 0.0000016755039767149002975769574342658*x1**6*x2**2 + 0.0022935503580488372709701168056536*x1**6 + 0.0000000084770378408610811335679789223875*x1**5*x2**3 + 0.00066246630028376451417792942327259*x1**5*x2 - 0.000000090872738160405297060009532782255*x1**4*x2**4 + 0.0000587712257074670581397334495673*x1**4*x2**2 - 0.043293162236620342030779825555*x1**4 + 0.000000048284186762233833650054905751692*x1**3*x2**5 - 0.00000070693399884904665290885692796379*x1**3*x2**3 - 0.011498721971926318237042992902235*x1**3*x2 - 0.000000042220880993703541178778525213859*x1**2*x2**6 + 0.0000024993174736924599113249316433513*x1**2*x2**4 - 0.00062869374168708543749955332395284*x1**2*x2**2 + 1.3766040360257852881175279547738*x1**2 + 0.000000019070032453503035836501227979603*x1*x2**7 - 0.0000010268565695710830158962861455953*x1*x2**5 + 0.0000099899285265325336749039124924384*x1*x2**3 + 0.1328515321125741824744458220011*x1*x2 - 0.0000000171545267437991843328792785435*x2**8 + 0.00000088295176323236077244354601115009*x2**6 - 0.000017928349508919918322384839560953*x2**4 + 0.033164814719539552652809778191405*x2**2


##########################################

# optimal control u
u = 0.00013820264900718645354307886109475*x1**7 + 0.000033494293776367085887847434804536*x1**6*x2 - 0.00000024639063265082033439151176640571*x1**5*x2**2 - 0.006624548869438665985008044343553*x1**5 + 0.0000036311049938582555969520567647438*x1**4*x2**3 - 0.0011751514420618845509034552471564*x1**4*x2 - 0.0000024130253344463610578557208709167*x1**3*x2**4 + 0.000021106374138851507769317541301*x1**3*x2**2 + 0.11498610043205782149711432944126*x1**3 + 0.0000025335514463953691104677488882002*x1**2*x2**5 - 0.000099954268630181831619610286812179*x1**2*x2**3 + 0.012572884203593940273612396658837*x1**2*x2 - 0.0000013352715220100140573456210852572*x1*x2**6 + 0.000051348070718158126014988718330098*x1*x2**4 - 0.00029956324863565775922711056090887*x1*x2**2 - 1.3285120436195179145547865063803*x1 + 0.000001372850670970261834931900284307*x2**7 - 0.000052994584981328800667974786554065*x2**5 + 0.0007173277720848024094585560317353*x2**3 - 0.66329674208206918333743805730719*x2

##########################################

# print("u = ", u)

f = [x2, -4.*x2 + 19.6*sin(x1) + 40.*u]

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

print(f"Verification results for SGA.")

if (result): 
  print(f"Not a Lyapunov function. Found counterexample: ")
  print(result)
else:
  print("Satisfy conditions with eps = ", epsilon)
  print(V, f" is a Lyapunov function.")

print("Verification time =", t)