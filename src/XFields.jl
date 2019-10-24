module XFields

using  FFTW
using  LinearAlgebra
import LinearAlgebra: dot
import Base: +, -, *, ^, \, getindex, promote_rule, convert, show, inv

const module_dir  = joinpath(@__DIR__, "..") |> normpath

include("fourier_transforms.jl")
export rFourierTransform, cFourierTransform, FourierTransform, Transform
export plan, Grid, wavenumber, frequencies, pixels

include("rFFT.jl")
export rFFT, rFFTunitary

include("cFFT.jl")
export cFFT, cFFTunitary

include("abstract_fields.jl")
export XField, fielddata, DiagOp, squash

include("sfields.jl")
export Smap, Sfourier, Sfield

end # module
