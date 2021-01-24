using FINUFFT

using Test
using LinearAlgebra
using Random

Random.seed!(1)

nj = 10
nk = 11
ms = 12
mt = 13
mu = 14

## DOUBLE PRECISION
tol = 1e-15

# nonuniform data
x = 3*pi*(1.0 .- 2*rand(nj))
y = 3*pi*(1.0 .- 2*rand(nj))
z = 3*pi*(1.0 .- 2*rand(nj))
c = rand(nj) + 1im*rand(nj)
s = rand(nk)
t = rand(nk)
u = rand(nk)
f = rand(nk) + 1im*rand(nk)

# uniform data
F1D = rand(ms) + 1im*rand(ms)
F2D = rand(ms, mt) + 1im*rand(ms,mt)
F3D = rand(ms, mt, mu) + 1im*rand(ms,mt, mu)

## SINGLE PRECISION
tolf = 1f-6

# nonuniform data
xf = 3f0*pi*(1 .- 2*rand(Float32, nj))
yf = 3f0*pi*(1 .- 2*rand(Float32, nj))
zf = 3f0*pi*(1 .- 2*rand(Float32, nj))
cf = rand(Float32, nj) + 1im*rand(Float32, nj)
sf = rand(Float32, nk)
tf = rand(Float32, nk)
uf = rand(Float32, nk)
ff = rand(Float32, nk) + 1im*rand(Float32,nk)

# uniform data
F1Df = rand(Float32, ms) + 1im*rand(Float32, ms)
F2Df = rand(Float32, ms, mt) + 1im*rand(Float32, ms, mt)
F3Df = rand(Float32, ms, mt, mu) + 1im*rand(Float32, ms, mt, mu)


modevec(m) = -floor(m/2):floor((m-1)/2+1)
k1 = modevec(ms)
k2 = modevec(mt)
k3 = modevec(mu)

@testset "NUFFT" begin
    @testset "DOUBLE" begin
        ## 1D
        @testset "1D" begin
            # 1D1
            @testset "1D1" begin
                out = complex(zeros(ms))
                ref = complex(zeros(ms))
                for j=1:nj
                    for ss=1:ms
                        ref[ss] += c[j] * exp(1im*k1[ss]*x[j])
                    end
                end
                # Try this one with explicit opts struct
                opts = finufft_default_opts()
                opts.spread_kerpad = 0 # This should also work
                nufft1d1!(x, c, 1, tol, out, opts)
                relerr_1d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d1 < 1e-13
                # Different caller
                out2 = nufft1d1(x, c, 1, tol, ms)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14
            end
            
            # 1D2
            @testset "1D2" begin
                out = complex(zeros(nj))
                ref = complex(zeros(nj))
                for j=1:nj
                    for ss=1:ms
                        ref[j] += F1D[ss] * exp(1im*k1[ss]*x[j])
                    end
                end
                nufft1d2!(x, out, 1, tol, F1D)
                relerr_1d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2 < 1e-13
                out2 = nufft1d2(x, 1, tol, F1D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14
            end
            
            # 1D3
            @testset "1D3" begin
                out = complex(zeros(nk))
                ref = complex(zeros(nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*s[k]*x[j])
                    end
                end
                nufft1d3!(x,c,1,tol,s,out)
                relerr_1d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d3 < 1e-13
                out2 = nufft1d3(x,c,1,tol,s)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end
        end

        ## 2D
        @testset "2D" begin
            @testset "2D1" begin
                # 2D1
                out = complex(zeros(ms, mt))
                ref = complex(zeros(ms, mt))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[ss,tt] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                nufft2d1!(x, y, c, 1, tol, out)
                relerr_2d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1 < 1e-13
                out2 = nufft2d1(x, y, c, 1, tol, ms, mt)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end

            @testset "2D2" begin
                # 2D2
                out = complex(zeros(nj))
                ref = complex(zeros(nj))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[j] += F2D[ss, tt] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]))
                        end
                    end
                end
                nufft2d2!(x, y, out, 1, tol, F2D)
                relerr_2d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2 < 1e-13
                out2 = nufft2d2(x, y, 1, tol, F2D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end

            @testset "2D3" begin
                # 2D3
                out = complex(zeros(nk))
                ref = complex(zeros(nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]))
                    end
                end
                nufft2d3!(x,y,c,1,tol,s,t,out)
                relerr_2d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d3 < 1e-13
                out2 = nufft2d3(x,y,c,1,tol,s,t)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end        
        end

        ## 3D
        @testset "3D" begin
            @testset "3D1" begin
                # 3D1
                out = complex(zeros(ms, mt, mu))
                ref = complex(zeros(ms, mt, mu))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[ss,tt,uu] += c[j] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                            end
                        end
                    end
                end
                nufft3d1!(x, y, z, c, 1, tol, out)
                relerr_3d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d1 < 1e-13
                out2 = nufft3d1(x, y, z, c, 1, tol, ms, mt, mu)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end

            @testset "3D2" begin
                # 3D2
                out = complex(zeros(nj))
                ref = complex(zeros(nj))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[j] += F3D[ss, tt, uu] * exp(1im*(k1[ss]*x[j]+k2[tt]*y[j]+k3[uu]*z[j]))
                            end
                        end
                    end
                end
                nufft3d2!(x, y, z, out, 1, tol, F3D)
                relerr_3d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2 < 1e-13
                out2 = nufft3d2(x, y, z, 1, tol, F3D)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end

            @testset "3D3" begin
                # 3D3
                out = complex(zeros(nk))
                ref = complex(zeros(nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += c[j] * exp(1im*(s[k]*x[j]+t[k]*y[j]+u[k]*z[j]))
                    end
                end        
                nufft3d3!(x,y,z,c,1,tol,s,t,u,out)
                relerr_3d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d3 < 1e-13
                out2 = nufft3d3(x,y,z,c,1,tol,s,t,u)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-14            
            end        
        end
    end

    @testset "SINGLE" begin
        ## 1D
        @testset "1D" begin
            # 1D1
            @testset "1D1" begin
                out = complex(zeros(Float32, ms))
                ref = complex(zeros(Float32, ms))
                for j=1:nj
                    for ss=1:ms
                        ref[ss] += cf[j] * exp(1im*k1[ss]*xf[j])
                    end
                end
                # Try this one with explicit opts struct
                opts = finufft_default_opts()
                opts.spread_kerpad = 0 # This should also work
                nufft1d1!(xf, cf, 1, tolf, out, opts)
                relerr_1d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d1 < 1e-5
                # Different caller
                out2 = nufft1d1(xf, cf, 1, tolf, ms)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldifff < 1e-6
            end
            
            # 1D2
            @testset "1D2" begin
                out = complex(zeros(Float32, nj))
                ref = complex(zeros(Float32, nj))
                for j=1:nj
                    for ss=1:ms
                        ref[j] += F1Df[ss] * exp(1im*k1[ss]*xf[j])
                    end
                end
                nufft1d2!(xf, out, 1, tolf, F1Df)
                relerr_1d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d2 < 1e-5
                out2 = nufft1d2(xf, 1, tolf, F1Df)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6
            end
            
            # 1D3
            @testset "1D3" begin
                out = complex(zeros(Float32, nk))
                ref = complex(zeros(Float32, nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += cf[j] * exp(1im*sf[k]*xf[j])
                    end
                end
                nufft1d3!(xf,cf,1,tolf,sf,out)
                relerr_1d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_1d3 < 1e-5
                out2 = nufft1d3(xf,cf,1,tolf,sf)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end
        end

        ## 2D
        @testset "2D" begin
            @testset "2D1" begin
                # 2D1
                out = complex(zeros(Float32, ms, mt))
                ref = complex(zeros(Float32, ms, mt))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[ss,tt] += cf[j] * exp(1im*(k1[ss]*xf[j]+k2[tt]*yf[j]))
                        end
                    end
                end
                nufft2d1!(xf, yf, cf, 1, tolf, out)
                relerr_2d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d1 < 1e-5
                out2 = nufft2d1(xf, yf, cf, 1, tolf, ms, mt)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end

            @testset "2D2" begin
                # 2D2
                out = complex(zeros(Float32, nj))
                ref = complex(zeros(Float32, nj))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            ref[j] += F2Df[ss, tt] * exp(1im*(k1[ss]*xf[j]+k2[tt]*yf[j]))
                        end
                    end
                end
                nufft2d2!(xf, yf, out, 1, tolf, F2Df)
                relerr_2d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d2 < 1e-5
                out2 = nufft2d2(xf, yf, 1, tolf, F2Df)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end

            @testset "2D3" begin
                # 2D3
                out = complex(zeros(Float32, nk))
                ref = complex(zeros(Float32, nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += cf[j] * exp(1im*(sf[k]*xf[j]+tf[k]*yf[j]))
                    end
                end
                nufft2d3!(xf,yf,cf,1,tolf,sf,tf,out)
                relerr_2d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_2d3 < 1e-5
                out2 = nufft2d3(xf,yf,cf,1,tolf,sf,tf)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end        
        end

        ## 3D
        @testset "3D" begin
            @testset "3D1" begin
                # 3D1
                out = complex(zeros(Float32, ms, mt, mu))
                ref = complex(zeros(Float32, ms, mt, mu))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[ss,tt,uu] += cf[j] * exp(1im*(k1[ss]*xf[j]+k2[tt]*yf[j]+k3[uu]*zf[j]))
                            end
                        end
                    end
                end
                nufft3d1!(xf, yf, zf, cf, 1, tolf, out)
                relerr_3d1 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d1 < 1e-5
                out2 = nufft3d1(xf, yf, zf, cf, 1, tolf, ms, mt, mu)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end

            @testset "3D2" begin
                # 3D2
                out = complex(zeros(Float32, nj))
                ref = complex(zeros(Float32, nj))
                for j=1:nj
                    for ss=1:ms
                        for tt=1:mt
                            for uu=1:mu
                                ref[j] += F3Df[ss, tt, uu] * exp(1im*(k1[ss]*xf[j]+k2[tt]*yf[j]+k3[uu]*zf[j]))
                            end
                        end
                    end
                end
                nufft3d2!(xf, yf, zf, out, 1, tolf, F3Df)
                relerr_3d2 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d2 < 1e-5
                out2 = nufft3d2(xf, yf, zf, 1, tolf, F3Df)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end

            @testset "3D3" begin
                # 3D3
                out = complex(zeros(Float32, nk))
                ref = complex(zeros(Float32, nk))
                for k=1:nk
                    for j=1:nj
                        ref[k] += cf[j] * exp(1im*(sf[k]*xf[j]+tf[k]*yf[j]+uf[k]*zf[j]))
                    end
                end        
                nufft3d3!(xf,yf,zf,cf,1,tolf,sf,tf,uf,out)
                relerr_3d3 = norm(vec(out)-vec(ref), Inf) / norm(vec(ref), Inf)
                @test relerr_3d3 < 1e-5
                out2 = nufft3d3(xf,yf,zf,cf,1,tolf,sf,tf,uf)
                reldiff = norm(vec(out)-vec(out2), Inf) / norm(vec(out), Inf)
                @test reldiff < 1e-6            
            end        
        end
    end
end
