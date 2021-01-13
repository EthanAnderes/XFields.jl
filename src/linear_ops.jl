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




# DiagOp <: AbstractLinearOp
# ======================================

# Most of the behavior of DiagOps is defined 
# using field ops ... in analogy to f(UΛU⁻¹) = U f(Λ) U⁻¹ 
# where f(Λ) ≡ Diagonal(f.(diag(Λ))) 

struct DiagOp{X<:Field} <: AbstractLinearOp
    f::X
end

# Interface methods
# ---------------------------------------

function Base.:inv(O::DiagOp{X}) where X<:Field 
	## DiagOp(X(fieldtransform(O.f), nan2zero.(inv.(fielddata(O.f)))))
	DiagOp(X(fieldtransform(O.f), pinv.(fielddata(O.f))))
end

function LinearAlgebra.adjoint(O::DiagOp{X}) where X<:Field 
	DiagOp(X(fieldtransform(O.f), conj.(fielddata(O.f))))
end

function Base.:*(O::DiagOp{X}, f::Y) where {F<:Transform, X<:Field{F}, Y<:Field{F}} 
	Y(O.f * X(f))
end

function Base.:\(O::DiagOp{X}, f::Y) where {F<:Transform, X<:Field{F}, Y<:Field{F}} 
	Y(O.f \ X(f))
end

_lmult(O::DiagOp{X}, f::Y) where {F<:Transform, X<:Field{F}, Y<:Field{F}} = O.f * X(f)

# getindex and basic operator functionality
# ------------------------------------------

LinearAlgebra.diag(O::DiagOp) = O.f

function Base.getindex(O::DiagOp, i)  
	getindex(O.f, i) # indexing is propigated
end

# Operations op(DiagOp), op(DiagOp, Number) and  op(Number, DiagOp)
# ------------------------------------------

# op(DiagOp, Number) and op(Number, DiagOp)
Base.:*(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(a * O.f)
Base.:*(a::Number, O::DiagOp{X})  where X<:Field = DiagOp(a * O.f)

Base.:\(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(O.f \ a)
Base.:\(a::Number, O::DiagOp{X})  where X<:Field = DiagOp(a \ O.f)

Base.:/(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(O.f / a)
Base.:/(a::Number, O::DiagOp{X})  where X<:Field = DiagOp(a / O.f)

Base.:-(O::DiagOp{X})             where X<:Field = DiagOp(-O.f)

Base.:^(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(O.f^a) 
Base.:^(O::DiagOp{X}, a::Integer) where X<:Field = DiagOp(O.f^a)

# op(DiagOp)
Base.:sqrt(O::DiagOp{X}) where X<:Field = DiagOp(sqrt(O.f))


# Operations between DiagOp and UniformScaling
# ------------------------------------------

LinearAlgebra.:*(J::UniformScaling, O::DiagOp) = DiagOp(J.λ * O.f)
LinearAlgebra.:*(O::DiagOp, J::UniformScaling) = DiagOp(J.λ * O.f)
LinearAlgebra.:+(O::DiagOp, J::UniformScaling) = DiagOp(J.λ + O.f)
LinearAlgebra.:+(J::UniformScaling, O::DiagOp) = DiagOp(J.λ + O.f)
LinearAlgebra.:-(O::DiagOp, J::UniformScaling) = DiagOp(O.f - J.λ)
LinearAlgebra.:-(J::UniformScaling, O::DiagOp) = DiagOp(J.λ - O.f)


# Operations op(DiagOp, DiagOp)
# ------------------------------------------

# ops of the same type
Base.:+(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f + O2.f)
Base.:-(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f - O2.f)
Base.:*(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f * O2.f)
Base.:\(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f \ O2.f)
Base.:/(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f / O2.f)

