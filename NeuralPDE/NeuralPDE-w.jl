using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum
import Flux: flatten, params

@parameters t, x, N, L, ρ_hat, μ, c₀, τ, L, l,vₕ, k, m, ω, λ, γ
@variables v(..), ρ(..)
# ρ_hat=0.89;
m=1;
μ=1; #choose as we like
τ=1; #choose as we like 
# l=sqrt(μ*τ/ρ_hat);

N = 168; 
ρₕ = 0.168;
L=N/ρₕ; 
δρ₀ = 0.02;
δv₀ = 0.01;
vₕ = 5.0461*((1+exp((ρₕ-0.25)/0.06))^-1 - 3.72*10^-6);

# vhat(ρ)= 5.0461*((1+exp((ρ-0.25)/0.06))^-1 - 3.72*10^-6);
# using Roots
# find_zero(vhat, (-5,5))
# 1.0001069901803379

# ρₕ=N/L;
k=2π/L;

c₀= 1.8634; 
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# δρₛ(x) = δρ₀*exp(complex(0,1)*k*x);
λ=k^2*c₀^2/100
ω=k*(vₕ+c₀)
γ=complex(λ,ω)

δρ(t,x)=δρ₀*exp(complex(0,k*x))*exp(-γ*t)
δv(t,x)=δv₀*exp(complex(0,k*x))*exp(-γ*t)

# Only real part
δρᵣ(t,x)=δρ₀*cos(k*x)*cos(ω*t)exp(-λ*t)
δvᵣ(t,x)=δv₀*cos(k*x)*cos(ω*t)exp(-λ*t)

#2D PDE
eqs  = [Dt(v(t,x)) + v(t,x)*Dx(v(t,x)) - (μ/ρ(t,x))*Dxx(v(t,x)) + (c₀^2/ρ(t,x))*Dx(ρ(t,x)) - (5.0461*((1+exp((ρ(t,x)-0.25)/0.06))^-1 - 3.72*10^-6) - v(t,x))/τ ~ 0,
        Dt(ρ(t,x)) + Dx(ρ(t,x)*v(t,x)) ~ 0]

# Initial and boundary conditions
# Initial and boundary conditions
bcs = [ρ(t,0) ~ ρ(t,L),
       v(t,0) ~ v(t,L),
       Dt(v(t,0)) ~ Dt(v(t,L)),
       # max(ρ(t,x)) ~ ρₕ,
       ρ(0,x) ~ ρₕ + δρᵣ(0,x),
       v(0,x) ~ vₕ + δvᵣ(0,x)]

# Space and time domains
domains = [t ∈ Interval(0.0,3000.0),
           x ∈ Interval(0.0,L)]

# Discretization
dx = 0.1

# Neural network
input_ = length(domains)
n = 15
# Neural network
dim = 2 # number of dimensions
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pde_system = PDESystem(eqs,bcs,domains,[t,x],[v,ρ])

prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=200)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=200)
phi = discretization.phi

using Plots

# Neural network
input_ = length(domains)
n = 5
chain =[FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:2]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
flat_initθ = reduce(vcat,initθ)

eltypeθ = eltype(initθ[1])
parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])
phi = NeuralPDE.get_phi.(chain,parameterless_type_θ)

map(phi_ -> phi_(rand(2,10), flat_initθ),phi)

derivative = NeuralPDE.get_numeric_derivative()

# :tangle neuralPDE.jl
indvars = [t,x]
depvars = [v,ρ]
dim = length(domains)
quadrature_strategy = NeuralPDE.QuadratureTraining()


_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,
                                                     chain,initθ,quadrature_strategy) for eq in  eqs]

map(loss_f -> loss_f(rand(2,10), flat_initθ),_pde_loss_functions)

bc_indvars = NeuralPDE.get_argument(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, phi, derivative,
                                                    chain,initθ,quadrature_strategy,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
map(loss_f -> loss_f(rand(2,10), flat_initθ),_bc_loss_functions)

# dx = 0.1
# train_sets = NeuralPDE.generate_training_sets(domains,dx,eqs,bcs,eltypeθ,indvars,depvars)
# pde_train_set,bcs_train_set = train_sets
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,eqs,bcs,eltypeθ,indvars,depvars,quadrature_strategy)

plbs,pubs = pde_bounds
pde_loss_functions = [NeuralPDE.get_loss_function(_loss,
                                                  lb,ub,
                                                  eltypeθ, parameterless_type_θ,
                                                  quadrature_strategy)
                      for (_loss,lb,ub) in zip(_pde_loss_functions, plbs,pubs)]

map(l->l(flat_initθ) ,pde_loss_functions)

blbs,bubs = bcs_bounds
bc_loss_functions = [NeuralPDE.get_loss_function(_loss,lb,ub,
                                                 eltypeθ, parameterless_type_θ,
                                                 quadrature_strategy)
                     for (_loss,lb,ub) in zip(_bc_loss_functions, blbs,bubs)]

map(l->l(flat_initθ) ,bc_loss_functions)

loss_functions =  [pde_loss_functions;bc_loss_functions]

function loss_function(θ,p)
    sum(map(l->l(θ) ,loss_functions))
end

f_ = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)

cb_ = function (p,l)
    println("loss: ", l )
    println("pde losses: ", map(l -> l(p), loss_functions[1:2]))
    println("bcs losses: ", map(l -> l(p), loss_functions[3:end]))
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb_, maxiters=20)

# using Plots

ts,xs = [infimum(d.domain):1:supremum(d.domain) for d in domains]

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

u_predict  = [[phi[i]([t,x],minimizers_[i])[1] for t in ts for x in xs] for i in 1:2]
# u_predict = [first(Array(phi([t, x], res.minimizer))) for t in ts for x in xs] 

for i in 1:2
    p1 = plot(ts, xs, u_predict[i],linetype=:contourf,title = "predict$i");
    plot(p1)
    savefig("sol_variable_corrected_bcs3$i")
end
