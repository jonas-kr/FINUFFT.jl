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
export nufft_opts
export nufft_c_opts # backward-compability

## External dependencies
using finufft_jll

const libfinufft = finufft_jll.libfinufft

const BIGINT = Int64 # defined in src/finufft.h

# FFTW floating-point types: (from FFTW.jl)
const fftwNumber = Union{Float64,Float32,Complex{Float64},Complex{Float32}}
const fftwReal = Union{Float64,Float32}
const fftwComplex = Union{Complex{Float64},Complex{Float32}}
const fftwDouble = Union{Float64,Complex{Float64}}
const fftwSingle = Union{Float32,Complex{Float32}}
const fftwTypeDouble = Union{Type{Float64},Type{Complex{Float64}}}
const fftwTypeSingle = Union{Type{Float32},Type{Complex{Float32}}}


## FINUFFT opts struct from src/finufft.h
"""
    mutable struct nufft_opts    
        modeord             :: Cint
        chkbnds             :: Cint              
        # 
        debug               :: Cint                
        spread_debug        :: Cint         
        showwarn            :: Cint
        # 
        nthreads            :: Cint
        fftw                :: Cint                 
        spread_sort         :: Cint          
        spread_kerevalmeth  :: Cint   
        spread_kerpad       :: Cint        
        upsampfac           :: Cdouble         
        spread_thread       :: Cint
        maxbatchsize        :: Cint
        spread_nthr_atomic  :: Cint
        spread_max_sp_size  :: Cint
    end

Options struct passed to the FINUFFT library.

# Fields

modeord :: Cint 
    (type 1,2 only):    0: CMCL-style increasing mode order
                        1: FFT-style mode order
chkbnds :: Cint
    0: don't check NU pts in [-3pi,3pi)
    1: do (<few % slower)

# diagnostic opts...
debug :: Cint
    0: silent
    1: some timing/debug
    2: more
spread_debug :: Cint
    0: silent
    1: some timing/debug
    2: tonnes
showwarn :: Cint
    0: don't print warnings to stderr
    1: do

# algorithm performance opts...
nthreads :: Cint
    number of threads to use, or 0 uses all available
fftw :: Cint
    plan flags to FFTW (FFTW_ESTIMATE=64, FFTW_MEASURE=0,...)
spread_sort :: Cint
    0: don't sort
    1: do
    2: heuristic choice
spread_kerevalmeth :: Cint
    0: exp(sqrt()) spreading kernel
    1: Horner piecewise poly (faster)
spread_kerpad :: Cint
    option only for exp(sqrt())
    0: don't pad kernel to 4n
    1: do
upsampfac :: Cdouble
    upsampling ratio sigma: 2.0 std, 1.25 small FFT, 0.0 auto
spread_thread :: Cint
    (vectorized ntr>1 only):    0: auto, 1: seq multithreaded,
                                2: parallel single-thread spread
maxbatchsize :: Cint
    option for vectorized ntr>1 only
    max transform batch, 0 auto
spread_nthr_atomic :: Cint
    if >=0, threads above which spreader OMP critical goes atomic
spread_max_sp_size :: Cint
    if >0, overrides spreader (dir=1) max subproblem size

"""
mutable struct nufft_opts    
    modeord             :: Cint
    chkbnds             :: Cint              
    # 
    debug               :: Cint                
    spread_debug        :: Cint         
    showwarn            :: Cint
    # 
    nthreads            :: Cint
    fftw                :: Cint                 
    spread_sort         :: Cint          
    spread_kerevalmeth  :: Cint   
    spread_kerpad       :: Cint        
    upsampfac           :: Cdouble         
    spread_thread       :: Cint
    maxbatchsize        :: Cint
    spread_nthr_atomic  :: Cint
    spread_max_sp_size  :: Cint
end

const nufft_c_opts = nufft_opts # backward compability

"""
    finufft_default_opts()

Return a [`nufft_opts`](@ref) struct with the default FINUFFT settings.\\
See: <https://finufft.readthedocs.io/en/latest/usage.html#options>
"""
function finufft_default_opts()
    opts = nufft_opts(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    ccall( (:finufft_default_opts, libfinufft),
           Nothing,
           (Ref{nufft_opts},),
           opts
           )
    # default to number of julia threads
    opts.nthreads = Threads.nthreads()
    return opts
end

### Error handling
const WARN_EPS_TOO_SMALL            = 1
const ERR_MAXNALLOC                 = 2
const ERR_SPREAD_BOX_SMALL          = 3
const ERR_SPREAD_PTS_OUT_RANGE      = 4
const ERR_SPREAD_ALLOC              = 5
const ERR_SPREAD_DIR                = 6
const ERR_UPSAMPFAC_TOO_SMALL       = 7
const HORNER_WRONG_BETA             = 8
const ERR_NDATA_NOTVALID            = 9
const ERR_TYPE_NOTVALID             = 10
# some generic internal allocation failure...
const ERR_ALLOC                     = 11
const ERR_DIM_NOTVALID              = 12
const ERR_SPREAD_THREAD_NOTVALID    = 13

struct FINUFFTError <: Exception
    errno::Cint
    msg::String
end
Base.showerror(io::IO, e::FINUFFTError) = print(io, "FINUFFT Error ($(e.errno)): ", e.msg)

function check_ret(ret)
    # Check return value and output error messages
    if ret==0
        return
    elseif ret==WARN_EPS_TOO_SMALL
        @warn "requested tolerance epsilon too small to achieve"
        return
    elseif ret==ERR_MAXNALLOC
        msg = "attemped to allocate internal array larger than MAX_NF (defined in defs.h)"
    elseif ret==ERR_SPREAD_BOX_SMALL
        msg = "spreader: fine grid too small compared to spread (kernel) width"
    elseif ret==ERR_SPREAD_PTS_OUT_RANGE
        msg = "spreader: if chkbnds=1, a nonuniform point coordinate is out of input range [-3pi,3pi]^d"
    elseif ret==ERR_SPREAD_ALLOC
        msg = "spreader: array allocation error"
    elseif ret==ERR_SPREAD_DIR
        msg = "spreader: illegal direction (should be 1 or 2)"
    elseif ret==ERR_UPSAMPFAC_TOO_SMALL
        msg = "upsampfac too small (should be >1.0)"
    elseif ret==HORNER_WRONG_BETA
        msg = "upsampfac not a value with known Horner poly eval rule (currently 2.0 or 1.25 only)"
    elseif ret==ERR_NDATA_NOTVALID
        msg = "ntrans not valid in many (vectorized) or guru interface (should be >= 1)"
    elseif ret==ERR_TYPE_NOTVALID
        msg = "transform type invalid"
    elseif ret==ERR_ALLOC
        msg = "general allocation failure"
    elseif ret==ERR_DIM_NOTVALID
        msg = "dimension invalid"
    elseif ret==ERR_SPREAD_THREAD_NOTVALID
        msg = "spread_thread option invalid"
    else
        msg = "unknown error"
    end
    throw(FINUFFTError(ret, msg))
end

### Simple Interfaces (allocate output)

## Type-1

"""
    nufft1d1(xj      :: Array{Float64}, 
             cj      :: Array{ComplexF64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             ms      :: Integer
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-1 1D complex nonuniform FFT. 
"""
function nufft1d1(xj::Array{T},
                  cj::Array{Complex{T}},
                  iflag::Integer,
                  eps::T,
                  ms::Integer,
                  opts::nufft_opts=finufft_default_opts()) where T <: fftwReal
    fk = Array{Complex{T}}(undef, ms)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-1 2D complex nonuniform FFT.
"""
function nufft2d1(xj      :: Array{T}, 
                  yj      :: Array{T}, 
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  ms      :: Integer,
                  mt      :: Integer,                   
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    fk = Array{Complex{T}}(undef, ms, mt)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-1 3D complex nonuniform FFT.
"""
function nufft3d1(xj      :: Array{T}, 
                  yj      :: Array{T},
                  zj      :: Array{T},                   
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  ms      :: Integer,
                  mt      :: Integer,
                  mu      :: Integer,                                     
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    fk = Array{Complex{T}}(undef, ms, mt, mu)
    nufft3d1!(xj, yj, zj, cj, iflag, eps, fk, opts)
    return fk
end


## Type-2

"""
    nufft1d2(xj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64} 
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-2 1D complex nonuniform FFT. 
"""
function nufft1d2(xj      :: Array{T},                    
                  iflag   :: Integer, 
                  eps     :: T,
                  fk      :: Array{Complex{T}},
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    cj = Array{Complex{T}}(undef, nj)
    nufft1d2!(xj, cj, iflag, eps, fk, opts)
    return cj
end

"""
    nufft2d2(xj      :: Array{Float64}, 
             yj      :: Array{Float64}, 
             iflag   :: Integer, 
             eps     :: Float64,
             fk      :: Array{ComplexF64} 
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-2 2D complex nonuniform FFT. 
"""
function nufft2d2(xj      :: Array{T}, 
                  yj      :: Array{T}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  fk      :: Array{Complex{T}},
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    cj = Array{Complex{T}}(undef, nj)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-2 3D complex nonuniform FFT. 
"""
function nufft3d2(xj      :: Array{T}, 
                  yj      :: Array{T},
                  zj      :: Array{T}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  fk      :: Array{Complex{T}},
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    cj = Array{Complex{T}}(undef, nj)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-3 1D complex nonuniform FFT.
"""
function nufft1d3(xj      :: Array{T}, 
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T},
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{Complex{T}}(undef, nk)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-3 2D complex nonuniform FFT.
"""
function nufft2d3(xj      :: Array{T},
                  yj      :: Array{T}, 
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T},
                  tk      :: Array{T},                  
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{Complex{T}}(undef, nk)
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
             [, opts :: nufft_opts]
            ) -> Array{ComplexF64}

Compute type-3 3D complex nonuniform FFT.
"""
function nufft3d3(xj      :: Array{T},
                  yj      :: Array{T},
                  zj      :: Array{T},                   
                  cj      :: Array{Complex{T}}, 
                  iflag   :: Integer, 
                  eps     :: T,
                  sk      :: Array{T},
                  tk      :: Array{T},
                  uk      :: Array{T},                  
                  opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    fk = Array{Complex{T}}(undef, nk)
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
              [, opts :: nufft_opts]
            )

Compute type-1 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d1!(xj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj) 
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d1(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
    # 	       CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft1d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, ms, fk, opts
                    )
    else
        ret = ccall( (:finufftf1d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, ms, fk, opts
        )
    end
    check_ret(ret)
end


"""
    nufft1d2!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_opts]
            )

Compute type-2 1D complex nonuniform FFT. Output stored in cj.
"""
function nufft1d2!(xj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(cj)==nj        
    ms = length(fk)    
    # Calling interface
    # int finufft1d2(BIGINT nj,FLT* xj,CPX* cj,int iflag,FLT eps,BIGINT ms,
    #                CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft1d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, ms, fk, opts
                    )
    else
        ret = ccall( (:finufftf1d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, ms, fk, opts
                    )
    end
    check_ret(ret)    
end


"""
    nufft1d3!(xj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              sk      :: Array{Float64},
              fk      :: Array{ComplexF64},
              [, opts :: nufft_opts]
             )

Compute type-3 1D complex nonuniform FFT. Output stored in fk.
"""
function nufft1d3!(xj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   sk      :: Array{T},
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(cj)==nj        
    nk = length(sk)
    @assert length(fk)==nk
    # Calling interface
    # int finufft1d3(BIGINT nj,FLT* x,CPX* c,int iflag,FLT eps,BIGINT nk, FLT* s, CPX* f, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft1d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, nk, sk, fk, opts
                    )
    else
        ret = ccall( (:finufftf1d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, cj, iflag, eps, nk, sk, fk, opts
                    )
    end
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
              [, opts :: nufft_opts]
            )

Compute type-1 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d1(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
    #                BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft2d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                    )
    else
        ret = ccall( (:finufftf2d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                    )
    end
    check_ret(ret)
end


"""
    nufft2d2!(xj      :: Array{Float64}, 
              yj      :: Array{Float64}, 
              cj      :: Array{ComplexF64}, 
              iflag   :: Integer, 
              eps     :: Float64,
              fk      :: Array{ComplexF64} 
              [, opts :: nufft_opts]
            )

Compute type-2 2D complex nonuniform FFT. Output stored in cj.
"""
function nufft2d2!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj    
    ms, mt = size(fk)    
    # Calling interface
    # int finufft2d2(BIGINT nj,FLT* xj,FLT *yj,CPX* cj,int iflag,FLT eps,
    #                BIGINT ms, BIGINT mt, CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft2d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                    )
    else
        ret = ccall( (:finufftf2d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, ms, mt, fk, opts
                    )
    end
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
              [, opts :: nufft_opts]
             )

Compute type-3 2D complex nonuniform FFT. Output stored in fk.
"""
function nufft2d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: Float64,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(fk)==nk    
    # Calling interface
    # int finufft2d3(BIGINT nj,FLT* x,FLT *y,CPX* cj,int iflag,FLT eps,BIGINT nk, FLT* s, FLT* t, CPX* fk, nufft_opts opts);    
    if T <: fftwDouble
        ret = ccall( (:finufft2d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, nk, sk, tk, fk, opts
                    )
    else
        ret = ccall( (:finufftf2d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, cj, iflag, eps, nk, sk, tk, fk, opts
                    )
    end
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
              [, opts :: nufft_opts]
            )

Compute type-1 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d1!(xj      :: Array{T}, 
                   yj      :: Array{T}, 
                   zj      :: Array{T}, 
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d1(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
    # 	       BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft3d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                    )
    else
        ret = ccall( (:finufftf3d1, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                    )
    end
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
              [, opts :: nufft_opts]
            )

Compute type-2 3D complex nonuniform FFT. Output stored in cj.
"""
function nufft3d2!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                    
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj    
    ms, mt, mu = size(fk)    
    # Calling interface
    # int finufft3d2(BIGINT nj,FLT* xj,FLT *yj,FLT *zj,CPX* cj,int iflag,FLT eps,
    #                BIGINT ms, BIGINT mt, BIGINT mu, CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft3d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                    )
    else
        ret = ccall( (:finufftf3d2, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},            
                    Ref{T},            
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    BIGINT,
                    BIGINT,
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, opts
                    )
    end
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
              [, opts :: nufft_opts]
             )

Compute type-3 3D complex nonuniform FFT. Output stored in fk.
"""
function nufft3d3!(xj      :: Array{T}, 
                   yj      :: Array{T},
                   zj      :: Array{T},                   
                   cj      :: Array{Complex{T}}, 
                   iflag   :: Integer, 
                   eps     :: T,
                   sk      :: Array{T},
                   tk      :: Array{T},
                   uk      :: Array{T},
                   fk      :: Array{Complex{T}},
                   opts    :: nufft_opts = finufft_default_opts()) where T <: fftwReal
    nj = length(xj)
    @assert length(yj)==nj
    @assert length(zj)==nj    
    @assert length(cj)==nj
    nk = length(sk)
    @assert length(tk)==nk
    @assert length(uk)==nk    
    @assert length(fk)==nk    
    # Calling interface
    # int finufft3d3(BIGINT nj,FLT* x,FLT *y,FLT *z, CPX* cj,int iflag,
    #                FLT eps,BIGINT nk,FLT* s, FLT* t, FLT *u,
    #                CPX* fk, nufft_opts opts);
    if T <: fftwDouble
        ret = ccall( (:finufft3d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                  
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                        
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, nk, sk, tk, uk, fk, opts
                    )
    else
        ret = ccall( (:finufftf3d3, libfinufft),
                    Cint,
                    (BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                  
                    Ref{Complex{T}},
                    Cint,
                    T,
                    BIGINT,
                    Ref{T},
                    Ref{T},
                    Ref{T},                        
                    Ref{Complex{T}},
                    Ref{nufft_opts}),
                    nj, xj, yj, zj, cj, iflag, eps, nk, sk, tk, uk, fk, opts
                    )
    end

    check_ret(ret)
end

end # module
