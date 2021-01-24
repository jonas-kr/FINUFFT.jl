include("FINUFFT.jl")

function main(T=Float64)
    nj = 100
    x = pi .* (1 .- rand(T,nj))
    x = collect(range(T(-pi/2),T(pi/2),length=nj))
    c = rand(Complex{T},nj)

    # Parameters
    ms = 20 # Output size
    tol = T(1e-6) # Tolerance

    # Output as return value
    fk = FINUFFT.nufft1d1(x, c, 1, tol, ms)
end