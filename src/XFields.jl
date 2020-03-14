module XFields

using  FFTW
using  LinearAlgebra
import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath
const F64   = Float64
const CF64  = Complex{Float64}

include("fourier_transforms.jl")
export rFourierTransform, cFourierTransform, FourierTransform, Transform
export plan, cplan, rplan
export Grid, wavenumber, frequencies, pixels
export adjoint

include("rFFT.jl")
export rFFT, rFFTunitary, get_rFFTimpulses

include("cFFT.jl")
export cFFT, cFFTunitary

include("abstract_fields.jl")
export XField, fielddata, DiagOp, diag

include("sfields.jl")
export Smap, Sfourier, Sfield
export dot

end # module
