nx= 100000;
# ν=0.4;
# c=5;
δx = 1000/(nx - 1);
x_range = range(0, stop=δx*(nx-1), length=nx) # Full range of spatial steps for wich a solution is desired

endTime = 100   # simulation end time
nt = 1000          # nt is the number of timesteps we want to calculate
δt = endTime/nt  # Δt is the amount of time each timestep covers
t = range(0, stop=endTime, length=nt) # Full range of time steps for which a solution is desired

function dif_nt(v, n)
    return v[n+1] - v[n]
end

function dif2_nt(v, n)
    return v[n+1] - 2*v[n] + v[n-1]
end

function mdif_nt(u,v,n)
    return u[n+1]*v[n] + v[n+1]*u[n] - 2*v[n]*u[n]
end

# u[:,] = copy(u_zero) # Initialise arbitrary future timestep with inital condition, u_zero_values

function kerner(v,ρ,Δx,Δt,params)
    N = length(v)
    vl=similar(v)
    ρl=similar(ρ)
    μ, c₀, τ = params
    N = length(ρ)
    k=2π/1000
    δv₀ = 0.01
    δρ₀ = 0.02
    
    V(ρ) = 5.0461*((1+exp((ρ-0.25)/0.06))^-1 - 3.72*10^-6) 

    for n in 2:N-1
        ρl[n] = ρ[n] - (Δt/Δx)*(mdif_nt(v,ρ,n))
        vl[n] = v[n] - (v[n]*Δt/Δx)*dif_nt(v,n) + (μ*Δt/(ρ[n]*(Δx)^2))*(dif2_nt(v,n)) + (c₀^2*Δt/ρ[n]*Δx)*(dif_nt(ρ,n)) + (Δt/τ)*(V(ρ[n])-v[n])
    end

    # Bondary condition
    # https://www.youtube.com/watch?v=uf4g_U8Ok3c&list=PLP8iPy9hna6Q2Kr16aWPOKE0dz9OnsnIJ&index=50&t=10m14s

    ## Fixed (Real parts of δρ e δv)
    # δρᵣ(x,t)=δρ₀*cos(k*x)*cos(ω*t)exp(-λ*t)
    δρᵣx(x)=δρ₀*cos(k*x)
    # δvᵣ(x,t)=δv₀*cos(k*x)*cos(ω*t)exp(-λ*t)
    δvᵣx(x)=δv₀*cos(k*x)
    # ρ[n,0] = ρₕ + δρᵣ(n,0)
    # ρ[n,0] = vₕ + δvᵣ(n,0)
    ρₕ = 0.168
    vₕ = 5.0461*((1+exp((ρₕ-0.25)/0.06))^-1 - 3.72*10^-6)
    ρl[length(ρ)] = ρₕ + δρᵣx(length(ρ)*Δx)
    vl[length(v)] = vₕ + δvᵣx(length(v)*Δx)
    ρl[1] = ρl[length(ρ)]
    vl[1] = vl[length(v)]
    # dif(v,0,i,arg=i) = dif(v,length(v),i,arg=i)

    return ρl, vl
end
# τ=1;
μ, c₀, τ = 1, 1.8634, 1 
params₀ = [μ, c₀, τ]

# Init array of ones at initial timestep
v₀ = ones(nx) 
ρ₀ = ones(nx) 

# # Set u₀ = 2 in the interval 0.5 ≤ x ≤ 1 as per our I.C.s
# v₀[0.5 .<= x .<= 100] .= 2  # Note use of . (dot) broadcasting syntax
# ρ₀[0.5 .<= x .<= 100] .= 5

kerner(v,ρ,Δx,Δt,V,params)
kerner(v₀, ρ₀,δx,δt,params₀)

