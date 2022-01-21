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

k=2π/L;

c₀= 1.8634; 
# δρₛ(x) = δρ₀*exp(complex(0,1)*k*x);
λ=k^2*c₀^2/100
ω=k*(vₕ+c₀)
γ=complex(λ,ω)

# Complete complex term
δρ(x,t)=δρ₀*exp(complex(0,k*x))*exp(-γ*t)
δv(x,t)=δv₀*exp(complex(0,k*x))*exp(-γ*t)
# Only real part
δρᵣ(x,t)=δρ₀*cos(k*x)*cos(ω*t)exp(-λ*t)
δvᵣ(x,t)=δv₀*cos(k*x)*cos(ω*t)exp(-λ*t)
