## =====================================================
abstract type XField{T<:Transform} end

@generated fielddata(x::XField) = :(tuple($((:(x.$f) for f=fieldnames(x))...)))
convert(::Type{X}, f::X) where X<:XField = f::X
@inline squash(x::T) where T<:Number = ifelse(isfinite(x), x, T(0))


## =====================================================
# operations with fields

# fields with scalars
(+)(f::F, n::Number) where F<:XField = F((ff .+ n for ff in fielddata(f))...)
(+)(n::Number, f::F) where F<:XField = F((ff .+ n for ff in fielddata(f))...)
(-)(f::F)            where F<:XField = F((.- ff for ff in fielddata(f))...)
(-)(f::F, n::Number) where F<:XField = F((ff .- n for ff in fielddata(f))...)
(-)(n::Number, f::F) where F<:XField = F((n .- ff for ff in fielddata(f))...)
(*)(f::F, n::Number) where F<:XField = F((n .*  ff for ff in fielddata(f))...)
(*)(n::Number, f::F) where F<:XField = F((n .*  ff for ff in fielddata(f))...)

# fields with UniformScaling
(*)(f::F, J::UniformScaling) where F<:XField = J.λ * f
(*)(J::UniformScaling, f::F) where F<:XField = J.λ * f

# op(fields, fields) which broadcast to the underlying fielddata for type similar args
for op in (:+, :-, :*)
    @eval ($op)(a::F, b::F) where {F<:XField} = F(map((a,b)->broadcast($op,a,b),fielddata(a),fielddata(b))...)
end

# op(fields, fields) operators for which we do automatic promotion
for op in (:+, :-)
    @eval ($op)(a::XField, b::XField) = ($op)(promote(a,b)...)
end


## =====================================================
# linear operators diagonal in a XField basis

struct DiagOp{F<:XField}
    f::F
end

# getindex and basic operator functionality
diag(O::DiagOp)         = O.f
getindex(O::DiagOp, i)  = getindex(O.f, i) # indexing is propigated
(*)(O::DiagOp{F}, f::G) where {F<:XField, G<:XField} = G(O.f * F(f))
(\)(O::DiagOp{F}, f::G) where {F<:XField, G<:XField} = G(inv(O).f * F(f))

# scalar ops with DiagOp
(*)(O::DiagOp{F}, a::Number)  where F<:XField = DiagOp(a * O.f)
(*)(a::Number, O::DiagOp{F})  where F<:XField = DiagOp(a * O.f)
(-)(O::DiagOp{F}) where F<:XField = DiagOp(-O.f)
(^)(op::DiagOp{F}, a::Number)  where F<:XField = DiagOp(F((i.^a for i in fielddata(op.f))...))
(^)(op::DiagOp{F}, a::Integer) where F<:XField = DiagOp(F((i.^a for i in fielddata(op.f))...))
sqrt(op::DiagOp{F}) where F<:XField            = DiagOp(F((sqrt.(i) for i in fielddata(op.f))...))
inv(op::DiagOp{F}) where F<:XField = DiagOp(F((squash.(inv.(i)) for i in fielddata(op.f))... ))

# ops of the same type
(+)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(O1.f + O2.f)
(-)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(O1.f - O2.f)
(*)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(O1.f * O2.f)
(\)(O1::DiagOp{F}, O2::DiagOp{F}) where F<:XField = DiagOp(inv(O1).f * O2.f)

# chains of linear ops that are not of the same type store a lazy tuple
(*)(O1::DiagOp, O2::DiagOp) = tuple(O1, O2)
(*)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(O1..., O2)
(*)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(O1, O2...)
(*)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(O1..., O2...)
(*)(O1::NTuple{N,DiagOp}, f::G) where {N,G<:XField} = foldr(*, (O1..., f))::G #::typeof(O1[1].f)

(\)(O1::DiagOp, O2::DiagOp)                   = tuple(inv(O1), O2)
(\)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(inv(O1)..., O2)
(\)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(inv(O1), O2...)
(\)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(inv(O1)..., O2...)
(\)(O1::NTuple{N,DiagOp}, f::G) where {N,G<:XField} = (inv(O1) * f)::G  #::typeof(O1[end].f)

inv(O1::NTuple{N,DiagOp}) where N = tuple((inv(op) for op in reverse(O1))...)

