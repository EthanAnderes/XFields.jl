module XFields

using Reexport
@reexport using LinearAlgebra
@reexport using FFTW

# using FFTransforms
# using Distributed
# using Statistics
# using PyCall
# using FFTW
# import FieldFlows
# using JLD2
# using ProgressMeter
# using FastTransforms
# using ApproxFun: Fun, Jacobi
# using Dierckx
# using HealpixHelper
# const HH = HealpixHelper
# const hp  = pyimport("healpy") 
# const one_K_in_mK = 1e+6
# FFTW.set_num_threads(4)

# using  FFTW
# using  LinearAlgebra
# import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
# import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath



#' Abstract Field and Transform pair
#' ==========================================

abstract type Transform{Tf<:Number,d} end

size_in(ft::Transform) = error("note yet defined")
size_out(ft::Transform) = error("note yet defined")
eltype_in(ft::Transform) = error("note yet defined")
eltype_out(ft::Transform) = error("note yet defined")
plan(ft::Transform) = error("note yet defined")
Ωx(ft::Transform) = error("note yet defined")



# plan(ft) * rand(eltype_in(ft), size_in(ft))
# plan(ft) \ rand(eltype_out(ft), size_out(ft))
# Ωx(F) -> for generating white noise (grid side of the form [1,2,3,4] just contributes 1* to the Ωx )




# Generally the main way to utalize this is to define a new
# transform type ...
#######
# struct 𝔽{Tf,d, ...} <: Transform{Tf,d}
#     ...
# end 
### with these defined 
#     In general with ft::𝔽{Tf,d}
#     size_in(ft)
#     size_out(ft)
#     eltype_in(ft)
#     eltype_out(ft)
#     plan(ft) * rand(eltype_in(ft), size_in(ft))
#     plan(ft) \ rand(eltype_out(ft), size_out(ft))




#' load src
#' ==========================================

abstract type Field{F<:Transform} end
abstract type FourierField{F<:Transform} <: Field{F} end
abstract type MapField{F<:Transform} <: Field{F} end
export Field, FourierField, MapField, Transform

include("xfield.jl")
export Xmap, Xfourier 

include("convert_promote.jl")

include("field_ops.jl")

include("linear_ops.jl")
export DiagOp, AbstractLinearOp






# I think it's possible to define this generically 
# so One can come up with new fields, for GPU arrays etc.
#######
# abstract type YField{F<:Transform,Tf,Ti,d} <: Field{F}  end

# struct Ymap{F<:Transform,Tf,Ti,d} <: YField{F,Tf,Ti,d}
#     ft::F 
#     f::??? something determined by ..Array{Tf,d}
#     function Xmap(ft::F, f::Array{Tf,d})  where {Tf,d,F<:𝔽{Tf,d}}
#         @assert size(f) == size_in(ft)
#         Ti = eltype_out(ft)
#         new{F,Tf,Ti,d}(ft, f)
#     end
# end

# • Rule for a Field (used in util.jl and field_ops.jl for fielddata and fieldtransfor) 
#   1st field of the struct is the transform
#   2nd field of the struct is the data 
#



#=
#' Quick field type
#' --------------------------------------

import Base: *, \

struct QUθφ{T,nθnφ}
    qu::Array{T,2}
end 

struct QUθk{T,nθnφ}
    qu::Array{Complex{T},2}
end

QUθφ{T,nθnφ}() where {T,nθnφ} = QUθφ{T,nθnφ}(zeros(T,nθnφ[1],nθnφ[2]))
QUθk{T,nθnφ}() where {T,nθnφ} = QUθφ{T,nθnφ}(zeros(Complex{T},nθnφ[1],nθnφ[2]÷2+1))
function Base.randn(::Type{QUθk{T,nθnφ}}) where {T,nθnφ} 
    nθ,nφ = nθnφ
    fqu = randn(Complex{T},nθ,nφ÷2+1)
    fqu[:,1] = randn(T,nθ,1)
    if iseven(nφ)
        fqu[:,end] = randn(T,nθ,1)
    end 
    return QUθk{T,nθnφ}(fqu)
end
Base.randn(::Type{QUθφ{T,nθnφ}}) where {T,nθnφ} = QUθφ{T,nθnφ}(randn(T,nθnφ[1], nθnφ[2]))

QUfield{T,nθnφ} = Union{QUθφ{T,nθnφ}, QUθk{T,nθnφ}}


#  container for planned FFT
struct θφ2θk{T,nθnφ} 
    FT::FFTW.rFFTWPlan{T,-1,false,2}
    IFT::FFTW.rFFTWPlan{Complex{T},1,false,2}
    normalize_FT::T
    normalize_IFT::T
end

# unitary by default
@generated function plan(::Type{θφ2θk{T,nθnφ}}) where {T,nθnφ}
    region = (2,)
    nθ, nφ = nθnφ
    X   = Array{T,2}(undef, nθ, nφ) 
    Y   = Array{Complex{T},2}(undef, FFTW.rfft_output_size(X, region)...)
    FT  = plan_rfft(X, region; flags=FFTW.ESTIMATE) 
    IFT = plan_brfft(FT*X, nφ, region; flags=FFTW.ESTIMATE) 

    normalize_FT  = 1/sqrt(nφ)
    normalize_IFT = FFTW.normalization(X, region) / normalize_FT

    return θφ2θk{T,nθnφ}(FT,IFT,normalize_FT, normalize_IFT)
end

Base.:*(p::θφ2θk, x::Array) = p.normalize_FT  .* (p.FT  * x)
Base.:\(p::θφ2θk, x::Array) = p.normalize_IFT .* (p.IFT * x)
Base.:*(p::θφ2θk{T,nθnφ}, f::QUθk{T,nθnφ}) where {T,nθnφ} = f
Base.:*(p::θφ2θk{T,nθnφ}, f::QUθφ{T,nθnφ}) where {T,nθnφ} = QUθk{T,nθnφ}(p * f.qu)
Base.:\(p::θφ2θk{T,nθnφ}, f::QUθk{T,nθnφ}) where {T,nθnφ} = QUθφ{T,nθnφ}(p \ f.qu)
Base.:\(p::θφ2θk{T,nθnφ}, f::QUθφ{T,nθnφ}) where {T,nθnφ} = f

Base.getindex(f::QUfield{T,nθnφ}, ::typeof(!)) where {T,nθnφ} = (plan(θφ2θk{T,nθnφ}) * f).qu
Base.getindex(f::QUfield{T,nθnφ}, ::Colon)     where {T,nθnφ} = (plan(θφ2θk{T,nθnφ}) \ f).qu

function Base.getindex(f::QUfield{T,nθnφ}, sym::Symbol) where {T,nθnφ}
    nθ, nφ = nθnφ
    p  = plan(θφ2θk{T,nθnφ})
    (sym == :qx) ? (p \ f).qu[1:nθ,:] :
    (sym == :ux) ? (p \ f).qu[(nθ+1):end,:] :
    (sym == :qk) ? (p * f).qu[1:nθ,:] :
    (sym == :uk) ? (p * f).qu[(nθ+1):end,:] :
    error("index is not defined")
end


=#





end
