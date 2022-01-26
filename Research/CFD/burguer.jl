nx=100;
# c=5;
delta_x = 40/(nx - 1)
x = range(0, stop=delta_x*(nx-1), length=nx) # Full range of spatial steps for wich a solution is desired

endTime = 20   # simulation end time
nt = 1000          # nt is the number of timesteps we want to calculate
delta_t = endTime/nt  # Δt is the amount of time each timestep covers
t = range(0, stop=endTime, length=nt) # Full range of time steps for which a solution is desired

# Init array of ones at initial timestep
u_zero = ones(nx)

# Set u₀ = 2 in the interval 0.5 ≤ x ≤ 1 as per our I.C.s
u_zero[0.5 .<= x .<= 10] .= 2  # Note use of . (dot) broadcasting syntax

u_zero

# u[:,] = copy(u_zero) # Initialise arbitrary future timestep with inital condition, u_zero
u=zeros((nx,nt+1))
u[:,1]=copy(u_zero)

for n in 1:nt       # loop over timesteps, n: nt times
    u[:,n+1] = copy(u[:,n]) # copy the existing values of u^n into u^(n+1)
    for i in 2:nx   # you can try commenting this line and...
        #for i in 1:nx    # ... uncommenting this line and see what happens!
        u[i,n+1] = u[i,n] - u[i,n] * delta_t/delta_x * (u[i,n] - u[i-1,n])
    end
end

using Plots; pyplot()
# gr() pyplot()

# anim = @animate for n in 1:10:nt
#     Plots.plot(x, u[:,n])
# end

# gif(anim, "gif_ploting.gif", fps=60)

xs = collect(x)
ts = collect(t)

plot(collect(x),collect(t),u'[1:1000,1:100],st=:surface, title="Burguer equation", xlabel="X", ylabel="Y", zlabel="U")
