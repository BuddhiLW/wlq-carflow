@variables x
Dx=Differential(x)

expand_derivatives(Dx(ϕ))

ν=0.4
ϕe = exp(-x^2/4*ν) + exp(-(x-2*π)^2/4*ν)

d(x)=exp(-0.1((x - 6.283185307179586)^2))*(1.2566370614359172 - (0.2x)) - (0.2x*exp(-0.1(x^2)))

(expand_derivatives(Dx(ϕe)))

first(substitute.(expand_derivatives(Dx(ϕe)), (Dict(x => 1),)))

dϕ(ξ) = first(substitute.(expand_derivatives(Dx(ϕe)), (Dict(x => ξ),)))

dϕ(1.78)

u_zero(x) = -2ν*(dϕ(x)/ϕ(x)) + 4

u_zero_values = map(ζ->u_zero(ζ), x_range)

Nx = 20
Lx = 1.0

deltax = Lx / Nx

xs = deltax/2:deltax:Lx

deltax = xs[2] - xs[1]

p2 = Plots.plot(0:0.001:Lx, u_zero, label="u₀", lw=1, ls=:dash)
Plots.scatter!(xs, u_zero.(xs), label="sampled")
Plots.scatter!(xs, zero.(xs), label="x nodes", alpha=0.5, ms=3, lw=2)

for i in 1:length(xs)
    plot!([ (xs[i] - deltax/2, u_zero(xs[i])), (xs[i] + deltax/2, u_zero(xs[i])) ], c=:green, lw=4, lab=false)

    plot!([ (xs[i] - deltax/2, 0), (xs[i] - deltax/2, u_zero(xs[i])), (xs[i] + deltax/2, u_zero(xs[i])), (xs[i] + deltax/2, 0)], c=:green, lw=1, lab=false, ls=:dash, alpha=0.3)
end
xlabel!("x")
ylabel!("u₀(x)")
Plots.savefig("./u0-burguer.png")

# u[:,] = copy(u_zero) # Initialise arbitrary future timestep with inital condition, u_zero_values

function burgers(u,δt,δx,ν)
    # u=zeros((nx,nt+1))
    N = length(u_zero_values)
    ul=copy(u) # start the u in a new time step.

    for i in 2:N-1
        ul[i] = u[i] + ν * δt/(δx)^2 *
            (u[i+1] - 2* u[i] + u[i-1])/2
    end

    # Bondary condition
    # https://www.youtube.com/watch?v=uf4g_U8Ok3c&list=PLP8iPy9hna6Q2Kr16aWPOKE0dz9OnsnIJ&index=50&t=10m14s
    ul[N] = u[N] + ν * δt/(δx)^2 *
        (u[1] - 2 * u[N] + u[N-1])/2

    ul[1] = u[1] + ν * δt/(δx)^2 *
        (u[2] - 2 * u[1] + u[N])/2

    return ul
end

burgers(u_zero_values,nt,nx,0.4)

function evolve(method, xs, δt, ν, t_final=10.0, f₀=u_zero)

    T = f₀.(xs)
    δx = xs[2] - xs[1]

    t = 0.0
    ts = [t]

    results = [T]

    while t < t_final
        Tl = method(T, δt, δx, ν)  # new
        push!(results, Tl)
        T = copy(Tl)

        t += δt
        push!(ts, t)
    end

    return ts, results
end

ts, results = evolve(burguers, x_range, δt, ν)

cc = "w l lc 'turquoise' lw 3 notitle"
F=plot(x_range, results[1], curveconf=cc);
for n in 1:10:nt
    wave = plot(x_range, results[n],
                lc=:turquoise,
                marker="ecircle",
                pn=7,
                ms=1.5,
                lw=3,
                grid = :on,
                w=:lp,
                legend = :Bessel_function)
    push!(F,wave)
end
save(term="gif", saveopts = "animate size 600,400 delay 1", output="./burguers.gif", handle=1)

cc = "w l lc 'turquoise' lw 3 notitle"
F=plot(x_range, results[1], curveconf=cc);
for n in 1:10:nt
    wave = plot(x_range, results[n],
                lc=:turquoise,
                marker="ecircle",
                pn=7,
                ms=1.5,
                lw=3,
                grid = :on,
                w=:lp,
                legend = :Bessel_function)
    push!(F,wave)
end
save(term="gif", saveopts = "animate size 600,400 delay 1", output="./burguers.gif", handle=1)

z=0:0.1:10pi;
step = 5;
cc = "w l lc 'turquoise' lw 3 notitle"
ac = Axes(zrange = (0,30), xrange = (-1.2, 1.2), yrange = (-1.2, 1.2),
          tics = :off,
          xlabel = :x, ylabel = :y, zlabel = :z)
F = scatter3(cos.(z[1:step]), sin.(z[1:step]), z[1:step], curveconf = cc, ac);
for i = 2:60
    pi = scatter3(cos.(z[1:i*step]), sin.(z[1:i*step]), z[1:i*step],
                  curveconf = cc, ac, handle = 2);
    push!(F, pi)
end
for i = 60:-1:1
    pi = scatter3(cos.(z[1:i*step]), sin.(z[1:i*step]), z[1:i*step],
                  curveconf = cc, ac, handle = 2);
    push!(F, pi)
end
save(term="gif", saveopts = "animate size 600,400 delay 1", output="anim3d.gif", handle=1)

gif(anim, "gif_ploting_burguer.gif", fps=60)
