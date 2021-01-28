# linear operators on fields
# ============================================================

"""
Interface requirements for AbstractLinearOp
```
Base.:inv(O::AbstractLinearOp) 
LinearAlgebra.adjoint(O::AbstractLinearOp) 
Base.:*(O::AbstractLinearOp, f::Field) 
Base.:\\(O::AbstractLinearOp, f::Field)
```
Optionally 
```
_lmult(O::AbstractLinearOp, f::Field) = O * f
```
can be specialized to avoid a final conversion back to 
the argument field type as `*` would typically do. 
Useful for a sequence of operations that avoid un-necessary conversions.
"""
abstract type AbstractLinearOp end


_lmult(O::AbstractLinearOp, f) = O * f


# Chains of AbstractLinearOp
# ------------------------------------------

# `*` concatinates chains
function Base.:*(O1::AbstractLinearOp, O2::AbstractLinearOp)
	tuple(O1, O2)
end

function Base.:*(O1::NTuple{N,AbstractLinearOp}, O2::AbstractLinearOp) where N 
	Base.front(O1) * (Base.last(O1) * O2)
end

function Base.:*(O1::AbstractLinearOp, O2::NTuple{N,AbstractLinearOp}) where N
	(O1 * Base.first(O2)) * Base.tail(O2)
end

function Base.:*(O1::NTuple{N,AbstractLinearOp}, O2::NTuple{M,AbstractLinearOp}) where {N,M}
	tuple(O1..., O2...)
end

# `inv` broadcasts and reverses order
function Base.:inv(O1::NTuple{N,AbstractLinearOp}) where N
	tuple((inv(op) for op in reverse(O1))...)
end

# `adjoint` broadcasts and reverses order
function Base.:adjoint(O1::NTuple{N,AbstractLinearOp}) where N
	tuple((adjoint(op) for op in reverse(O1))...)
end

# `\`
function Base.:\(O1::AbstractLinearOp, O2::AbstractLinearOp)
	inv(O1) * O2
end

function Base.:\(O1::NTuple{N,AbstractLinearOp}, O2::AbstractLinearOp) where N
	inv(O1) * O2
end

function Base.:\(O1::AbstractLinearOp, O2::NTuple{N,AbstractLinearOp}) where N
	inv(O1) * O2
end

function Base.:\(O1::NTuple{N,AbstractLinearOp}, O2::NTuple{M,AbstractLinearOp}) where {N,M}
	inv(O1) * O2
end

# `/`
function Base.:/(O1::AbstractLinearOp, O2::AbstractLinearOp) 
	O1 * inv(O2) 
end

function Base.:/(O1::NTuple{N,AbstractLinearOp}, O2::AbstractLinearOp) where N
	O1 * inv(O2)
end

function Base.:/(O1::AbstractLinearOp, O2::NTuple{N,AbstractLinearOp}) where N
	O1 * inv(O2)
end

function Base.:/(O1::NTuple{N,AbstractLinearOp}, O2::NTuple{M,AbstractLinearOp}) where {N,M}
	O1 * inv(O2)
end

# activate the lazy tuple when operating
function Base.:*(O1::NTuple{N,AbstractLinearOp}, f::Y) where {N,Y<:Field}
	Y(foldr(_lmult, (O1..., f)))
end

function Base.:\(O1::NTuple{N,AbstractLinearOp}, f::Y) where {N,Y<:Field}
	(inv(O1) * f)::Y
end 

