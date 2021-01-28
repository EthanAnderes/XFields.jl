# DiagOp and FourierOp both <: AbstractLinearOp
# ======================================

# DiagOp has a Field as storage and represents
# the diagonal of a matrix  in the corresponding transformation basis
# determined by the Field type.
struct DiagOp{X<:Field} <: AbstractLinearOp
    f::X
end

# Interface methods for DiagOp
# ---------------------------------------

function Base.:inv(O::DiagOp{X}) where X<:Field 
	DiagOp(X(fieldtransform(O.f), pinv.(fielddata(O.f))))
end

# This requires orthogonality of the transform
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

# indexing is propigated
Base.getindex(O::DiagOp, i) = getindex(O.f, i) 

# Operations op(DiagOp), op(DiagOp, Number) and  op(Number, DiagOp)
# ------------------------------------------

# op(DiagOp, Number) and op(Number, DiagOp)
Base.:*(O::DiagOp, a::Number) = DiagOp(a * O.f)
Base.:*(a::Number, O::DiagOp) = DiagOp(a * O.f)

Base.:\(O::DiagOp, a::Number)  = DiagOp(O.f \ a)
Base.:\(a::Number, O::DiagOp)  = DiagOp(a \ O.f)

Base.:/(O::DiagOp, a::Number)  = DiagOp(O.f / a)
Base.:/(a::Number, O::DiagOp)  = DiagOp(a / O.f)

Base.:-(O::DiagOp)             = DiagOp(-O.f)

Base.:^(O::DiagOp, a::Number)  = DiagOp(O.f^a) 
Base.:^(O::DiagOp, a::Integer) = DiagOp(O.f^a)

# op(DiagOp)
Base.:sqrt(O::DiagOp) = DiagOp(sqrt(O.f))


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
