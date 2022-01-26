nxx=100;
# c=5;
delta_xx = 10/(nxx - 1)
xx = range(0, stop=delta_xx*(nxx-1), length=nxx) # Fvll range of spatial steps for wich a solvtion is desired

endTime = 20   # simvlation end time
nt = 1000          # nt is the nvmber of timesteps we want to calcvlate
delta_t = endTime/nt  # Δt is the amovnt of time each timestep covers
t = range(0, stop=endTime, length=nt) # Fvll range of time steps for which a solvtion is desired

# Init array of ones at initial timestep
# v_zero = ones(nxx)

f(x) = (1/(2*√π))*exp((-1/2)*(x-3)^2)
# Set v₀ = 2 in the interval 0.5 ≤ xx ≤ 1 as per ovr I.C.s
v_zero = f.(xx)  # Note vse of . (dot) broadcasting syntaxx

v_zero

# v[:,] = copy(v_zero) # Initialise arbitrary fvtvre timestep with inital condition, v_zero
v=zeros((nxx,nt+1))
v[:,1]=copy(v_zero)

for n in 1:nt       # loop over timesteps, n: nt times
    v[:,n+1] = copy(v[:,n]) # copy the exxisting valves of v^n into v^(n+1)
    for i in 2:nxx   # yov can try commenting this line and...
        #for i in 1:nxx    # ... vncommenting this line and see what happens!
        v[i,n+1] = v[i,n] - v[i,n] * delta_t/delta_xx * (v[i,n] - v[i-1,n])
    end
end

using Plots; pyplot()

xxs = collect(xx)
ts = collect(t)

plot(collect(xx),collect(t),v'[1:1000,1:100],st=:surface, title="Burguer equation", xlabel="X", ylabel="Time", zlabel="V")
