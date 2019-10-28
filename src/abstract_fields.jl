abstract type XField{T<:Transform} end

@generated fielddata(x::XField) = :(tuple($((:(x.$f) for f=fieldnames(x))...)))
convert(::Type{X}, f::X) where X<:XField = f::X

## =========== operations with fields ===========

## fields with scalars
(+)(f::F, n::Number) where F<:XField = F((ff .+ n for ff in fielddata(f))...)
(+)(n::Number, f::F) where F<:XField = F((ff .+ n for ff in fielddata(f))...)
(-)(f::F)            where F<:XField = F((.- ff for ff in fielddata(f))...)
(-)(f::F, n::Number) where F<:XField = F((ff .- n for ff in fielddata(f))...)
(-)(n::Number, f::F) where F<:XField = F((n .- ff for ff in fielddata(f))...)
(*)(f::F, n::Number) where F<:XField = F((n .*  ff for ff in fielddata(f))...)
(*)(n::Number, f::F) where F<:XField = F((n .*  ff for ff in fielddata(f))...)

## fields with UniformScaling
(*)(f::F, J::UniformScaling) where F<:XField = J.λ * f
(*)(J::UniformScaling, f::F) where F<:XField = J.λ * f

## op(fields, fields) which broadcast to the underlying fielddata for type similar args
for op in (:+, :-, :*)
    @eval ($op)(a::F, b::F) where {F<:XField} = F(map((a,b)->broadcast($op,a,b),fielddata(a),fielddata(b))...)
end

## op(fields, fields) operators for which we do automatic promotion
for op in (:+, :-, :dot)
    @eval ($op)(a::XField, b::XField) = ($op)(promote(a,b)...)
end


## =========== linear operators diagonal in a XField basis===========

struct DiagOp{F<:XField}
    f::F
end

(*)(O::DiagOp{F}, f::XField) where F<:XField = O.f * F(f)
(\)(O::DiagOp{F}, f::XField) where F<:XField = inv(O) * f

(+)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(O1.f + O2.f)
(-)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(O1.f - O2.f)
(-)(O::DiagOp{F}) where F<:XField = DiagOp(-O.f)

(*)(O::DiagOp{F}, a::Number)  where F<:XField = DiagOp(a * O.f)
(*)(a::Number, O::DiagOp{F})  where F<:XField = DiagOp(a * O.f)

(^)(op::DiagOp{F}, a::Number)  where F<:XField = DiagOp(F((i.^a for i in fielddata(op.f))...))
(^)(op::DiagOp{F}, a::Integer) where F<:XField = DiagOp(F((i.^a for i in fielddata(op.f))...))
sqrt(op::DiagOp{F}) where F<:XField            = DiagOp(F((sqrt.(i) for i in fielddata(op.f))...))

inv(op::DiagOp{F}) where F<:XField = DiagOp(F((squash.(inv.(i)) for i in fielddata(op.f))... ))
@inline squash(x::T) where T = ifelse(isfinite(x), x, T(0))

# chains of linear ops with * and 
(*)(O1::DiagOp, O2::DiagOp)                   = tuple(O1, O2)
(*)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(O1..., O2)
(*)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(O1, O2...)
(*)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(O1..., O2...)
(*)(O1::NTuple{N,DiagOp}, f::XField)  where N = foldr(*, (O1..., f))::typeof(O1[1].f)
(\)(O1::DiagOp, O2::DiagOp)                   = tuple(inv(O1), O2)
(\)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(inv(O1)..., O2)
(\)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(inv(O1), O2...)
(\)(O1::NTuple{N,DiagOp}, f::XField) where N  = (inv(O1) * f)::typeof(O1[end].f)
inv(O1::NTuple{N,DiagOp}) where N             = tuple((inv(op) for op in reverse(O1))...)
(\)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(inv(O1)..., O2...)

