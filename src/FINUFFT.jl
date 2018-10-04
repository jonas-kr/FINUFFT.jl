__precompile__()
module FINUFFT

## Export
export nufft1d1, nufft1d2, nufft1d3
export nufft2d1, nufft2d2, nufft2d3
export nufft3d1, nufft3d2, nufft3d3

export nufft1d1!, nufft1d2!, nufft1d3!
export nufft2d1!, nufft2d2!, nufft2d3!
export nufft3d1!, nufft3d2!, nufft3d3!

export finufft_default_opts
export nufft_c_opts

## External dependencies
using Compat
using Compat.Libdl

const depsfile = joinpath(dirname(@__DIR__), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("FINUFFT is not properly installed. Please build it first.")
end

function __init__()
    Libdl.dlopen(fftw, Libdl.RTLD_GLOBAL)       
    Libdl.dlopen(fftw_threads, Libdl.RTLD_GLOBAL)
end

## FINUFFT opts struct from src/finufft_h.c
"""
    mutable struct nufft_c_opts

Options struct passed to the FINUFFT library.

# Fields

    debug :: Cint
0: silent, 1: text basic timing output

    spread_debug :: Cint
passed to spread_opts, 0 (no text) 1 (some) or 2 (lots)

    spread_sort :: Cint
passed to spread_opts, 0 (don't sort) 1 (do) or 2 (heuristic)

    spread_kerevalmeth :: Cint
passed to spread_opts, 0: exp(sqrt()), 1: Horner ppval (faster)

    spread_kerpad :: Cint
passed to spread_opts, 0: don't pad to mult of 4, 1: do

    chkbnds :: Cint
0: don't check if input NU pts in [-3pi,3pi], 1: do

    fftw :: Cint
0:`FFTW_ESTIMATE`, or 1:`FFTW_MEASURE` (slow plan but faster)

    modeord :: Cint
0: CMCL-style increasing mode ordering (neg to pos), or\\
1: FFT-style mode ordering (affects type-1,2 only)

    upsampfac::Cdouble
upsampling ratio sigma, either 2.0 (standard) or 1.25 (small FFT)
"""
mutable struct nufft_c_opts    
    debug::Cint                
    spread_debug::Cint         
    spread_sort::Cint          
    spread_kerevalmeth::Cint   
    spread_kerpad::Cint        
    chkbnds::Cint              
    fftw::Cint                 
    modeord::Cint                                             
    upsampfac::Cdouble         
end

"""
    finufft_default_opts()

Return a [`nufft_c_opts`](@ref) struct with the default FINUFFT settings.\\
See: https://finufft.readthedocs.io/en/latest/usage.html#options
"""
function finufft_default_opts()
    opts = nufft_c_opts(0,0,0,0,0,0,0,0,0)
    ccall( (:finufft_default_c_opts, libfinufft),
           Nothing,
           (Ref{nufft_c_opts},),
           opts
           )
    return opts
end

function check_ret(ret)
    # Check return value and output error messages
    if ret==0
        return
    elseif ret==1
        msg = "requested tolerance epsilon too small"
    elseif ret==2
        msg = "attemped to allocate internal arrays larger than MAX_NF (defined in common.h)"
    elseif ret==3
        msg = "spreader: fine grid too small"
    elseif ret==4
        msg = "spreader: if chkbnds=1, a nonuniform point out of input range [-3pi,3pi]^d"
    elseif ret==5
        msg = "spreader: array allocation error"
    elseif ret==6
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==7
        msg = "upsampfac too small (should be >1)"
    elseif ret==8
        msg = "upsampfac not a value with known Horner eval: currently 2.0 or 1.25 only"
    elseif ret==9
        msg = "ndata not valid (should be >= 1)"
    else
        msg = "unknown error"
    end
    error("FINUFFT error: $msg")
end

### Simple Interfaces (allocate output)

## Type-1

"""
    nufft1d1(xj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-1 1D complex nonuniform FFT. 
"""
function nufft1d1(xj::Array{Float64},
                  cj::Array{ComplexF64},
                  iflag::Integer,
                  eps::Float64,
                  ms::Integer,
                  opts::nufft_c_opts=finufft_default_opts())
    fk = Array{ComplexF64}(undef, ms)
    nufft1d1!(xj, cj, iflag, eps, fk, opts)
    return fk
end

"""
    nufft2d1(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer,
             mt      :: Integer,
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-1 2D complex nonuniform FFT.
"""
function nufft2d1(xj      :: Array{Float64}, 
                  yj      :: Array{Float64}, 
                  cj      :: Array{ComplexF64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  ms      :: Integer,
                  mt      :: Integer,                   
                  opts    :: nufft_c_opts = finufft_default_opts())
    fk = Array{ComplexF64}(undef, ms, mt)
    nufft2d1!(xj, yj, cj, iflag, eps, fk, opts)
    return fk
end

"""
    nufft3d1(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             zj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer,
             mt      :: Integer,
             mu      :: Integer,
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-1 3D complex nonuniform FFT.
"""
function nufft3d1(xj      :: Array{Float64}, 
                  yj      :: Array{Float64},
                  zj      :: Array{Float64},                   
                  cj      :: Array{ComplexF64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  ms      :: Integer,
                  mt      :: Integer,
                  mu      :: Integer,                                     
                  opts    :: nufft_c_opts = finufft_default_opts())
    fk = Array{ComplexF64}(undef, ms, mt, mu)
    nufft3d1!(xj, yj, zj, cj, iflag, eps, fk, opts)
    return fk
end


## Type-2

"""
    nufft1d2(xj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64} 
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-2 1D complex nonuniform FFT. 
"""
function nufft1d2(xj      :: Array{Float64},                    
                  iflag   :: Integer, 
                  eps     :: Float64,
                  fk      :: Array{ComplexF64},
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    cj = Array{ComplexF64}(undef, nj)
    nufft1d2!(xj, cj, iflag, eps, fk, opts)
    return cj
end

"""
    nufft2d2(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64} 
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-2 2D complex nonuniform FFT. 
"""
function nufft2d2(xj      :: Array{Float64}, 
                  yj      :: Array{Float64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  fk      :: Array{ComplexF64},
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    cj = Array{ComplexF64}(undef, nj)
    nufft2d2!(xj, yj, cj, iflag, eps, fk, opts)
    return cj
end

"""
    nufft3d2(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             zj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64} 
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-2 3D complex nonuniform FFT. 
"""
function nufft3d2(xj      :: Array{Float64}, 
                  yj      :: Array{Float64},
                  zj      :: Array{Float64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  fk      :: Array{ComplexF64},
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    cj = Array{ComplexF64}(undef, nj)
    nufft3d2!(xj, yj, zj, cj, iflag, eps, fk, opts)
    return cj
end


## Type-3

"""
    nufft1d3(xj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64},
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-3 1D complex nonuniform FFT.
"""
function nufft1d3(xj      :: Array{Float64}, 
                  cj      :: Array{ComplexF64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  sk      :: Array{Float64},
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{ComplexF64}(undef, nk)
    nufft1d3!(xj, cj, iflag, eps, sk, fk, opts);
    return fk
end

"""
    nufft2d3(xj      :: Array{Float64}, 
             yj      :: Array{Float64},
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64},
             tk      :: Array{Float64}
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-3 2D complex nonuniform FFT.
"""
function nufft2d3(xj      :: Array{Float64},
                  yj      :: Array{Float64}, 
                  cj      :: Array{ComplexF64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  sk      :: Array{Float64},
                  tk      :: Array{Float64},                  
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{ComplexF64}(undef, nk)
    nufft2d3!(xj, yj, cj, iflag, eps, sk, tk, fk, opts);
    return fk
end

"""
    nufft3d3(xj      :: Array{Float64}, 
             yj      :: Array{Float64},
             zj      :: Array{Float64},
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             sk      :: Array{Float64},
             tk      :: Array{Float64}
             uk      :: Array{Float64}
             [, opts :: nufft_c_opts]
            ) -> Array{ComplexF64}

Compute type-3 3D complex nonuniform FFT.
"""
function nufft3d3(xj      :: Array{Float64},
                  yj      :: Array{Float64},
                  zj      :: Array{Float64},                   
                  cj      :: Array{ComplexF64}, 
                  iflag   :: Integer, 
                  eps     :: Float64,
                  sk      :: Array{Float64},
                  tk      :: Array{Float64},
                  uk      :: Array{Float64},                  
                  opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{ComplexF64}(undef, nk)
    nufft3d3!(xj, yj, zj, cj, iflag, eps, sk, tk, uk, fk, opts);
    return fk
end


### Direct interfaces (No allocation)

## 1D

"""
    nufft1d1!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-1 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d1!(xj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj) 
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d1_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ret = ccall( (:finufft1d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)
end


"""
    nufft1d2!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-2 1D complex nonuniform FFT. Output stored in cj.
"""
function nufft1d2!(xj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d2_c(int nj,FLT* xj,FLT _Complex* cj,int iflag, FLT eps,int ms, FLT _Complex* fk, nufft_c_opts opts);
    ret = ccall( (:finufft1d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, ms, fk, opts
                 )
    check_ret(ret)    
end


"""
    nufft1d3!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              fk      :: Array{ComplexF64},
              [, opts :: nufft_c_opts]
             )

Compute type-3 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d3!(xj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   sk      :: Array{Float64},
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    @assert length(fk)==nk
    # Calling interface
    # int finufft1d3_c(int j,FLT* x,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT _Complex* f, nufft_c_opts opts);
    ret = ccall( (:finufft1d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, cj, iflag, eps, nk, sk, fk, opts
                 )
    check_ret(ret)
end


## 2D

"""
    nufft2d1!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-1 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d1!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d1_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt,FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft2d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end


"""
    nufft2d2!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-2 2D complex nonuniform FFT. Output stored in cj.
"""
function nufft2d2!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d2_c(int nj,FLT* xj,FLT *yj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft2d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft2d3!(xj      :: Array{Float64}, 
              yj      :: Array{Float64},
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              tk      :: Array{Float64},
              fk      :: Array{ComplexF64}
              [, opts :: nufft_c_opts]
             )

Compute type-3 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d3!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64},
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   sk      :: Array{Float64},
                   tk      :: Array{Float64},
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(fk)==nk    
    # Calling interface
    # iint finufft2d3_c(int nj,FLT* x,FLT *y,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT _Complex* f, nufft_c_opts copts);
    ret = ccall( (:finufft2d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, cj, iflag, eps, nk, sk, tk, fk, opts
                 )
    check_ret(ret)
end

## 3D

"""
    nufft3d1!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              zj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-1 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d1!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64}, 
                   zj      :: Array{Float64}, 
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d1_c(int nj,FLT* xj,FLT* yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu,FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft3d1_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft3d2!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              zj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_c_opts]
            )

Compute type-2 3D complex nonuniform FFT. Output stored in cj.
"""
function nufft3d2!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64},
                   zj      :: Array{Float64},                    
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d2_c(int nj,FLT* xj,FLT *yj,FLT *zj,FLT _Complex* cj,int iflag, FLT eps,int ms, int mt, int mu, FLT _Complex* fk, nufft_c_opts copts);
    ret = ccall( (:finufft3d2_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},            
                  Ref{Cdouble},            
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Cint,
                  Cint,
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                 )
    check_ret(ret)
end

"""
    nufft3d3!(xj      :: Array{Float64}, 
              yj      :: Array{Float64},
              zj      :: Array{Float64},
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              tk      :: Array{Float64},
              uk      :: Array{Float64},
              fk      :: Array{ComplexF64}
              [, opts :: nufft_c_opts]
             )

Compute type-3 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d3!(xj      :: Array{Float64}, 
                   yj      :: Array{Float64},
                   zj      :: Array{Float64},                   
                   cj      :: Array{ComplexF64}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   sk      :: Array{Float64},
                   tk      :: Array{Float64},
                   uk      :: Array{Float64},
                   fk      :: Array{ComplexF64},
                   opts    :: nufft_c_opts = finufft_default_opts())
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(uk)==nk    
    @assert length(fk)==nk    
    # Calling interface
    # int finufft3d3_c(int nj,FLT* x,FLT *y,FLT *z,FLT _Complex* c,int iflag,FLT eps,int nk, FLT* s, FLT *t,FLT *u,FLT _Complex* f, nufft_c_opts copts);
    ret = ccall( (:finufft3d3_c, libfinufft),
                 Cint,
                 (Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                  
                  Ref{ComplexF64},
                  Cint,
                  Cdouble,
                  Cint,
                  Ref{Cdouble},
                  Ref{Cdouble},
                  Ref{Cdouble},                        
                  Ref{ComplexF64},
                  nufft_c_opts),
                 nj, xj, yj, zj, cj, iflag, eps, nk, sk, tk, uk, fk, opts
                 )
    check_ret(ret)
end

end # module
