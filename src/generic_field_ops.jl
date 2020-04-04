
#%% util methods
#%% -----------------------------------------------------------

@generated function fielddata(x::XField)
	f = fieldname(x, 1)
	:(x.$f)
end

convert(::Type{X}, f::X) where X<:XField = f::X

nan2zero(x::T) where T<:Number = ifelse(isfinite(x), x, zero(T))
nan2zero(f::XF) where XF<:XField = XF(broadcast(nan2zero, fielddata(f)))

pos_part(x::T) where T<:Number = clamp(x,zero(T),T(Inf))
neg_part(x::T) where T<:Number = -clamp(x,-T(Inf),zero(T))

#%% field operations
#%% -----------------------------------------------------------

# op(f::XField, n::Number) and op(n::Number, f::XField)
for op in (:+, :-, :*)
	@eval ($op)(f::XF, n::Number) where XF<:XField = XF(broadcast($op, fielddata(f),n))
	@eval ($op)(n::Number, f::XF) where XF<:XField = XF(broadcast($op, n, fielddata(f)))
end

# op(f::XField)
for op in (:-, :sqrt, :inv)
	@eval ($op)(f::XF) where XF<:XField = XF(broadcast($op, fielddata(f)))
end

#%% op(f::XF, g::XF) for like types XF<:XField
for op in (:+, :-, :*)
    @eval ($op)(a::XF, b::XF) where {XF<:XField} = XF(broadcast($op, fielddata(a),fielddata(b)))
end

#%%  op(f::F, g::F) for different types F<:XField
for op in (:+, :-)
    @eval ($op)(a::XField, b::XField) = ($op)(promote(a,b)...)
end

#%% fields and UniformScaling
(*)(f::XF, J::UniformScaling) where XF<:XField = J.λ * f
(*)(J::UniformScaling, f::XF) where XF<:XField = J.λ * f
(\)(J::UniformScaling, f::XF) where XF<:XField = (1/J.λ) * f

#%% linear operators diagonal in a XField basis
#%% ============================================================

struct DiagOp{XF<:XField}
    f::XF
end

#%% getindex and basic operator functionality
#%% -----------------------------------------------------------

diag(O::DiagOp) = O.f
getindex(O::DiagOp, i)   = getindex(O.f, i) # indexing is propigated
(*)(O::DiagOp{XF}, f::XG) where {XF<:XField, XG<:XField} = XG(O.f * XF(f))
(\)(O::DiagOp{XF}, f::XG) where {XF<:XField, XG<:XField} = XG(inv(O).f * XF(f))

#%% Operations with
#%% -----------------------------------------------------------

# op(DiagOp, Number) and op(Number, DiagOp)
(*)(O::DiagOp{XF}, a::Number)  where XF<:XField = DiagOp(a * O.f)
(*)(a::Number, O::DiagOp{XF})  where XF<:XField = DiagOp(a * O.f)
(-)(O::DiagOp{XF})             where XF<:XField = DiagOp(-O.f)
(^)(O::DiagOp{XF}, a::Number)  where XF<:XField = DiagOp(O.f^a)
(^)(O::DiagOp{XF}, a::Integer) where XF<:XField = DiagOp(O.f^a)

# op(DiagOp)
sqrt(O::DiagOp{XF}) where XF<:XField = DiagOp(sqrt(O.f))
inv(O::DiagOp{XF})  where XF<:XField = DiagOp(XF(nan2zero.(inv.(fielddata(O.f)))))

# ops of the same type
(+)(O1::DiagOp{XF}, O2::DiagOp{XF}) where XF<:XField = DiagOp(O1.f + O2.f)
(-)(O1::DiagOp{XF}, O2::DiagOp{XF}) where XF<:XField = DiagOp(O1.f - O2.f)
(*)(O1::DiagOp{XF}, O2::DiagOp{XF}) where XF<:XField = DiagOp(O1.f * O2.f)
(\)(O1::DiagOp{XF}, O2::DiagOp{XF}) where XF<:XField = DiagOp(inv(O1).f * O2.f)

# chains of linear ops that are not of the same type store a lazy tuple
(*)(O1::DiagOp, O2::DiagOp) = tuple(O1, O2)
(*)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(O1..., O2)
(*)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(O1, O2...)
(*)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(O1..., O2...)
(*)(O1::NTuple{N,DiagOp}, f::XG) where {N,XG<:XField} = foldr(*, (O1..., f))::XG #::typeof(O1[1].f)

(\)(O1::DiagOp, O2::DiagOp)                   = tuple(inv(O1), O2)
(\)(O1::NTuple{N,DiagOp}, O2::DiagOp) where N = tuple(inv(O1)..., O2)
(\)(O1::DiagOp, O2::NTuple{N,DiagOp}) where N = tuple(inv(O1), O2...)
(\)(O1::NTuple{N,DiagOp}, O2::NTuple{M,DiagOp}) where {N,M} = tuple(inv(O1)..., O2...)
(\)(O1::NTuple{N,DiagOp}, f::XG) where {N,XG<:XField} = (inv(O1) * f)::XG  #::typeof(O1[end].f)

inv(O1::NTuple{N,DiagOp}) where N = tuple((inv(op) for op in reverse(O1))...)

#%% linear ops with UniformScaling
(*)(J::UniformScaling, O::DiagOp) = DiagOp(J.λ * O.f)
(*)(O::DiagOp, J::UniformScaling) = DiagOp(J.λ * O.f)

(+)(O::DiagOp, J::UniformScaling) = DiagOp(J.λ + O.f)
(+)(J::UniformScaling, O::DiagOp) = DiagOp(J.λ + O.f)
(-)(O::DiagOp, J::UniformScaling) = DiagOp(O.f - J.λ)
(-)(J::UniformScaling, O::DiagOp) = DiagOp(J.λ - O.f)


