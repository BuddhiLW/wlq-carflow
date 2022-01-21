using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

#V(ρ)=1.5*(1-ρ/2)²;
@parameters t, x, μ, c₀, τ, L
@variables v(..), ρ(..)
μ=0.3;
c₀= sqrt(5.5);
τ = 0.02;
L = 3.0;
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eqs  = [Dt(v(t,x)) + v(t,x)*Dx(v(t,x)) - (μ/ρ(t,x))*Dxx(v(t,x)) + (c₀^2/ρ(t,x))*Dx(ρ(t,x)) - (1.5*(1-ρ(t,x)/2)^2 - v(t,x))/τ ~ 0,
        Dt(ρ(t,x)) + Dx(ρ(t,x)*v(t,x)) ~ 0]

# Initial and boundary conditions
bcs = [ρ(t,0) ~ ρ(L,0),
       v(t,0) ~ v(t,L),
       Dt(v(t,0)) ~ Dt(v(t,L))]

# Space and time domains
domains = [t ∈ Interval(0.0,10.0),
           x ∈ Interval(0.0,L)]

# Discretization
dx = 0.05

# Neural network
input_ = length(domains)
n = 15
chain =[FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:3]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

_strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, _strategy, init_params= initθ)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[v,ρ])
prob = discretize(pde_system,discretization)
sym_prob = symbolic_discretize(pde_system,discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
bcs_inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

cb = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=5000)

phi = discretization.phi

using Plots

ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
v_predict_contourf = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs] ,length(xs),length(ts))
plot(ts, xs, u_predict_contourf, linetype=:contourf,title = "predict")

v_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ]
p1= plot(xs, v_predict[3],title = "t = 0.1");
p2= plot(xs, v_predict[11],title = "t = 0.5");
p3= plot(xs, v_predict[end],title = "t = 1");
plot(p1,p2,p3)

