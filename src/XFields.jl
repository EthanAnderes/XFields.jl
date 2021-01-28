module XFields

using LinearAlgebra

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# Abstract Transform and identity transform
# ==========================================
export Transform, size_in, size_out, eltype_in, eltype_out, plan

abstract type Transform{Tf<:Number,d} end

size_in(ft::Transform) = error("not yet defined")
size_out(ft::Transform) = error("not yet defined")
eltype_in(ft::Transform) = error("not yet defined")
eltype_out(ft::Transform) = error("not yet defined")
plan(ft::Transform) = error("not yet defined")

# Identity transform when one just wants to work with an array
struct Id{Tf,d} <: Transform{Tf,d}
	sz::NTuple{d,Int} 
end
size_in(trn::Id) = trn.sz
size_out(trn::Id) = trn.sz
eltype_in(trn::Id{Tf}) where {Tf} = Tf
eltype_out(trn::Id{Tf}) where {Tf} = Tf
plan(trn::Id) = trn
Base.:*(trn::Id{Tf,d}, f::Array{Tf,d}) where {Tf,d} = f
Base.:\(trn::Id{Tf,d}, f::Array{Tf,d}) where {Tf,d} = f

# Abstract Field with Fourier <-> Map pair
# ==========================================
export	Field, FourierField, MapField, fielddata, fieldtransform,
		AbstractLinearOp, DiagOp, diag 

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

include("abstract_linear_ops.jl")

include("diag_linear_op.jl")

# Specific implimentation of Abstract Field 
# =========================================
export Xmap, Xfourier, Xfield 

include("xfield.jl")


end
