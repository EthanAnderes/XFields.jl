module XFields

using  FFTW
using  LinearAlgebra
import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath


#%% Transforms abstract type structure
#%% ============================================================

abstract type Transform{T,d} end
abstract type SliceTransform{T,dn,dt} end
# These correspond to field storage types Array{T, d} and Array{T, dn+dt} in map field

abstract type r2cTransform{T<:Real,dnᵢ,nᵢ}  <: Transform{T,dnᵢ} end
abstract type c2cTransform{T<:Real,dnᵢ,nᵢ}  <: Transform{T,dnᵢ} end
# These add nᵢ where nᵢ == size(map.x)

abstract type r2cSliceTransform{T<:Real,dnᵢ,dtᵢ,nᵢ,tᵢ}  <: r2cTransform{T,dnᵢ,dtᵢ} end
abstract type c2cSliceTransform{T<:Real,dnᵢ,dtᵢ,nᵢ,tᵢ}  <: c2cTransform{T,dnᵢ,dtᵢ} end
# These add nᵢ,tᵢ where tuple(nᵢ...,tᵢ...) == size(map.x)

export Transform, SliceTransform, r2cTransform, c2cTransform
export r2cSliceTransform, c2cSliceTransform


#%% XField abstract type
#%% ============================================================

abstract type XField{F<:Transform} end
export XField


#%% 
#%% ============================================================

include("generic_field_ops.jl")
export fielddata, DiagOp, diag

include("fourier_transforms/fourier_transforms.jl")
export r2cFourierTransform, c2cFourierTransform, FourierTransform
export plan, cplan, rplan
export Grid, wavenumber, frequencies, pixels
export adjoint

include("fourier_transforms/rFFT.jl")
export rFFT, rFFTunitary, get_rFFTimpulses

include("fourier_transforms/cFFT.jl")
export cFFT, cFFTunitary

include("rfields.jl")
export Rmap, Rfourier, Rfield
export dot

include("cfields.jl")
export Cmap, Cfourier, Cfield
export dot


end # module
