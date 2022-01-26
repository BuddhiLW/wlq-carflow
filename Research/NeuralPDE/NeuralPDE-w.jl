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

N = 10; # 168
ρₕ = 0.10; # 0.168
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
λ=(k^2*c₀^2)/100
ω=k*(vₕ+c₀)
γ=complex(λ,ω)

# δρ(t,x)=δρ₀*exp(complex(0,k*x))*exp(-γ*t)
# δv(t,x)=δv₀*exp(complex(0,k*x))*exp(-γ*t)

# Only real part
δρᵣ(t,x)=δρ₀*cos(k*x)*cos(ω*t)exp(-λ*t)
δvᵣ(t,x)=δv₀*cos(k*x)*cos(ω*t)exp(-λ*t)

#2D PDE
eqs  = [Dt(v(t,x)) + v(t,x)*Dx(v(t,x)) - (μ/ρ(t,x))*Dxx(v(t,x)) + (c₀^2/ρ(t,x))*Dx(ρ(t,x)) - (5.0461*((1 + exp(((ρ(t,x)-0.25)/0.06)))^-1 - 3.72*10^-2) - v(t,x))/τ ~ 0,
        Dt(ρ(t,x)) + Dx(ρ(t,x)*v(t,x)) ~ 0]
# Initial and boundary conditions
bcs = [ρ(t,0) ~ ρ(t,L),
       v(t,0) ~ v(t,L),
       Dx(v(t,0)) ~ Dx(v(t,L)),
       Dt(v(t,0)) ~ Dt(v(t,L)),
       ρ(0,x) ~ ρₕ + δρᵣ(0,x),
       v(0,x) ~ vₕ + δvᵣ(0,x)]

# Space and time domains
domains = [t ∈ Interval(0.0,1000.0),
           x ∈ Interval(0.0,L)]

# # Discretization
# dx = 0.1

# # Neural network
# dim = 2 # number of dimensions
# chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
# discretization = PhysicsInformedNN(chain, QuadratureTraining())
# @named pde_system = PDESystem(eqs,bcs,domains,[t,x],[v,ρ])

# prob = discretize(pde_system,discretization)

# cb = function (p,l)
#     println("Current loss is: $l")
#     return false
# end

# opt = Optim.BFGS()
# res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=200)
# prob = remake(prob,u0=res.minimizer)
# res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=200)
# phi = discretization.phi

# using Plots

# ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]

# u_predict = reshape([first(phi([t,x], res.minimizer)) for t in ts for x in xs], (length(ts), length(xs)))
# plot(ts[1:3001],xs[1:3001],u_predict[1:3001,1:3001], st=:surface, title="Fist 1000 t.u., velocity")
# # for i in 1:2
# #     # :contourf
# #     p1 = plot(ts, xs, u_predict[i],st=:surface,title = "predict$i");
# #     plot(p1)
# #     savefig("sol_variable_20220124_$i")
# # end



# ##########
# ##########    GIF
# u_predict = [first(Array(phi([t, x], res.minimizer))) for t in ts for x in xs]
# using Plots
# using Printf

# function plot_(res)
#     anim = @animate for (i, t) in enumerate(0:1:10)
#         @info "Animating frame $i..."
#         u_predict_anime = reshape([Array(phi([t, x], res.minimizer))[1] for x in xs], length(xs))
#         title = @sprintf("predict, t = %.3f", t)
#         p1 = plot(ts, xs, u_predict_anime,st=:surface, label="", title=title)
#         plot(p1)
#     end
#     gif(anim,"3pde-2.gif", fps=10)
# end

# https://www.overleaf.com/project/61eeef5f10e3884a2684742a









######## Try
# u_predict = [first(Array(phi([t, x], res.minimizer))) for t in ts for x in xs]
# Neural network
input_ = length(domains)
n = 15
chain =[FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:2]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

_strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, _strategy, init_params= initθ)

@named pde_system = PDESystem(eqs,bcs,domains,[t,x],[v(t,x),ρ(t,x)])
# @named pde_system = PDESystem(eqs,bcs,domains,[t,x],[u1(t, x),u2(t, x)])
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

res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=100) #5000
phi = discretization.phi

ts,xs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

u_predict  = [[phi[i]([t,x],minimizers_[i])[1] for t in ts  for x in xs] for i in 1:2]

for i in 1:2
    p2 = plot(ts, xs, u_predict[i],linetype=:surface,title = "predict");
    plot(p2)
    savefig("sol_u$i")
end

# using Plots
# using Printf

# function plot_(res)
#     anim = @animate for (i, t) in enumerate(0:1:10) #1000
#         @info "Animating frame $i..."
#         u_predict_v = reshape([Array(phi([t, x], res.minimizer))[1] for x in xs], length(xs))
#         # u_predict_pho = reshape([Array(phi([t, x], res.minimizer))[2] for x in xs], length(xs))
#         title = @sprintf("predict, t = %.3f", t)
#         p1 = plot(xs, u_predict_v,st=:surface, label="", title=title)
#         plot(p1)
#     end
#     gif(anim,"3pde.gif", fps=10)
# end

# plot_(res)


using Plots
using Printf

function plot_(res)
    anim = @animate for (i, t) in enumerate(0:1:10)
        @info "Animating frame $i..."
        u_v = reshape([Array(phi([t, x], minimizers_[1]))[1] for x in xs], length(xs))
        u_pho = reshape([Array(phi([t, x], minimizers_[1]))[1] for x in xs], length(xs))
        title = @sprintf("predict, t = %.3f", t)
        p1 = plot(ts, xs, u_v,st=:surface, label="", title=title)
        p2 = plot(ts, xs, u_pho,st=:surface, label="", title=title)
        plot(p1,p2)
    end
    gif(anim,"3pde-2.gif", fps=10)
end
