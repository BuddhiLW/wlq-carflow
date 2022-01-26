using Pkg;
Pkg.add("Symbolics")
Pkg.add("Gaston")
Pkg.add("SpecialFunctions")

using Symbolics
# using Plots
using Gaston, SpecialFunctions
# using GR

nx= 50;
ν=0.4;
# c=5;
δx = 15/(nx - 1);
x_range = range(0, stop=δx*(nx-1), length=nx) # Full range of spatial steps for wich a solution is desired

endTime = 100   # simulation end time
nt = 1000          # nt is the number of timesteps we want to calculate
δt = endTime/nt  # Δt is the amount of time each timestep covers
t = range(0, stop=endTime, length=nt) # Full range of time steps for which a solution is desired

ν=0.4;
ϕ(x) = exp(-x^2/4*ν) + exp(-(x-2*π)^2/4*ν)

ϕ(1.1)
