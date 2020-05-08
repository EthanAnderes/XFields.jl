module XFields

using Reexport
using LinearAlgebra
using FFTW

# import Base: +, -, *, ^, \, sqrt, getindex, promote_rule, convert, show, inv, transpose
# import LinearAlgebra: dot, adjoint, diag, \

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# Abstract Transform
# ==========================================

abstract type Transform{Tf<:Number,d} end

size_in(ft::Transform) = error("not yet defined")
size_out(ft::Transform) = error("not yet defined")
eltype_in(ft::Transform) = error("not yet defined")
eltype_out(ft::Transform) = error("not yet defined")
plan(ft::Transform) = error("not yet defined")

export Transform, size_in, size_out, eltype_in, eltype_out, plan

# Abstract Field with Fourier <-> Map pair
# ==========================================

abstract type Field{F<:Transform,Tf,Ti,d} end
abstract type FourierField{F<:Transform,Tf,Ti,d} <: Field{F,Tf,Ti,d} end
abstract type MapField{F<:Transform,Tf,Ti,d} <: Field{F,Tf,Ti,d} end

FourierField(::Type{X}) where {X<:Field} = error("not yet defined, should return the dual type")
MapField(::Type{X})     where {X<:Field} = error("not yet defined, should return the dual type")
# note:  FourierField(f::Field) and MapField(f::Field) 
# fall back to convert using the above definitions

fieldtransform(x::Field) = error("not yet defined")
fielddata(x::Field)      = error("not yet defined")

include("convert_promote.jl")

include("field_methods.jl")

include("linear_ops.jl")

export Field, FourierField, MapField, DiagOp, AbstractLinearOp, fielddata, fieldtransform

# Specific implimentation 
# =========================================

include("xfield.jl")

export Xmap, Xfourier, Xfield 

end
