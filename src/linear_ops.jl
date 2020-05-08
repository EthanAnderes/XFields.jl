# linear operators diagonal in a Field basis
# ============================================================

abstract type AbstractLinearOp{X<:Field} end

struct DiagOp{X<:Field} <: AbstractLinearOp{X}
    f::X
end

# Most of the behavior of DiagOps is defined 
# using field ops ... in analogy to foo(UΛUᵀ) = U foo(Λ) Uᵀ 
# where foo(Λ) can be considered an operation of the field diag(Λ)

# getindex and basic operator functionality
# ------------------------------------------

LinearAlgebra.diag(O::DiagOp) = O.f
Base.getindex(O::DiagOp, i)   = getindex(O.f, i) # indexing is propigated
Base.:*(O::DiagOp{X}, f::Y) where {F<:Transform, X<:Field{F}, Y<:Field{F}} = Y(O.f * X(f))
Base.:\(O::DiagOp{X}, f::Y) where {F<:Transform, X<:Field{F}, Y<:Field{F}} = Y(inv(O).f * X(f))

# Operations with
# ------------------------------------------

# op(DiagOp, Number) and op(Number, DiagOp)
Base.:*(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(a * O.f)
Base.:*(a::Number, O::DiagOp{X})  where X<:Field = DiagOp(a * O.f)
Base.:-(O::DiagOp{X})             where X<:Field = DiagOp(-O.f)
Base.:^(O::DiagOp{X}, a::Number)  where X<:Field = DiagOp(O.f^a) # Where does this get defined??
Base.:^(O::DiagOp{X}, a::Integer) where X<:Field = DiagOp(O.f^a)

# op(DiagOp)
Base.:sqrt(O::DiagOp{X}) where X<:Field = DiagOp(sqrt(O.f))
Base.:inv(O::DiagOp{X})  where X<:Field = DiagOp(X(fieldtransform(O.f), nan2zero.(inv.(fielddata(O.f)))))

# ops of the same type
Base.:+(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f + O2.f)
Base.:-(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f - O2.f)
Base.:*(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(O1.f * O2.f)
Base.:\(O1::DiagOp{X}, O2::DiagOp{X}) where X<:Field = DiagOp(inv(O1).f * O2.f)

# chains of linear ops that are not of the same type store a lazy tuple
Base.:*(O1::DiagOp, O2::DiagOp) = tuple(O1, O2)
Base.:*(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(O1..., O2)
Base.:*(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(O1, O2...)
Base.:*(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(O1..., O2...)
Base.:*(O1::NTuple{N,DiagOp}, f::Y) where {N,Y<:Field} = foldr(*, (O1..., f))::Y #::typeof(O1[1].f)

Base.:\(O1::DiagOp, O2::DiagOp)                   = tuple(inv(O1), O2)
Base.:\(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(inv(O1)..., O2)
Base.:\(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(inv(O1), O2...)
Base.:\(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(inv(O1)..., O2...)
Base.:\(O1::NTuple{N,DiagOp}, f::Y) where {N,Y<:Field} = (inv(O1) * f)::Y  #::typeof(O1[end].f)

Base.:inv(O1::NTuple{N,DiagOp}) where N = tuple((inv(op) for op in reverse(O1))...)

# linear ops with UniformScaling
LinearAlgebra.:*(J::UniformScaling, O::DiagOp) = DiagOp(J.λ * O.f)
LinearAlgebra.:*(O::DiagOp, J::UniformScaling) = DiagOp(J.λ * O.f)
LinearAlgebra.:+(O::DiagOp, J::UniformScaling) = DiagOp(J.λ + O.f)
LinearAlgebra.:+(J::UniformScaling, O::DiagOp) = DiagOp(J.λ + O.f)
LinearAlgebra.:-(O::DiagOp, J::UniformScaling) = DiagOp(O.f - J.λ)
LinearAlgebra.:-(J::UniformScaling, O::DiagOp) = DiagOp(J.λ - O.f)






