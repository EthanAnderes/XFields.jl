module XFields

using Reexport
using LinearAlgebra
using FFTW

# import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
# import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# Abstract Field and Transform pair
# ==========================================

abstract type Transform{Tf<:Number,d} end

size_in(ft::Transform) = error("note yet defined")
size_out(ft::Transform) = error("note yet defined")
eltype_in(ft::Transform) = error("note yet defined")
eltype_out(ft::Transform) = error("note yet defined")
plan(ft::Transform) = error("note yet defined")

export Transform, size_in, size_out, eltype_in, eltype_out, plan

# load src
# ==========================================

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

end
