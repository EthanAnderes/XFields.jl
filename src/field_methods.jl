# getindex for :, ! and *, / with the transform 
# ============================================================

#  getindex
Base.getindex(f::Field, ::typeof(!)) = fielddata(FourierField(f))
Base.getindex(f::Field, ::Colon)     = fielddata(MapField(f))

# convert to the corresponding dual field using the transform itself
Base.:*(ft::F, f::Field{F,Tf,Ti,d})  where {Tf,Ti,d, F<:Transform{Tf,d}} = FourierField(f)
Base.:\(ft::F, f::Field{F,Tf,Ti,d})  where {Tf,Ti,d, F<:Transform{Tf,d}} = MapField(f)


# field operations
# ============================================================

# op(f::Field, n::Number) and op(n::Number, f::Field)
for op in (:+, :-, :*)
    @eval Base.$op(f::X, n::Number) where X<:Field = X(fieldtransform(f), broadcast($op, fielddata(f),n))
    @eval Base.$op(n::Number, f::X) where X<:Field = X(fieldtransform(f), broadcast($op, n, fielddata(f)))
end
Base.:^(f::X, n::Number)  where X<:Field  = X(fieldtransform(f), broadcast(^, fielddata(f),n))
Base.:^(f::X, n::Integer) where X<:Field = X(fieldtransform(f), broadcast(^, fielddata(f),n))

# op(f::Field)
for op in (:-, :sqrt, :inv)
    @eval Base.$op(f::X) where X<:Field = X(fieldtransform(f), broadcast($op, fielddata(f)))
end

nan2zero(x::T) where T<:Number = ifelse(isfinite(x), x, zero(T))
pos_part(x::T) where T<:Number = clamp(x,zero(T),T(Inf))
neg_part(x::T) where T<:Number = -clamp(x,-T(Inf),zero(T))

nan2zero(f::X) where X<:Field = X(fieldtransform(f), broadcast(nan2zero, fielddata(f)))
pos_part(f::X) where X<:Field = X(fieldtransform(f), broadcast(pos_part, fielddata(f)))
neg_part(f::X) where X<:Field = X(fieldtransform(f), broadcast(neg_part, fielddata(f)))

# op(f::X, g::X) for like types X<:Field
for op in (:+, :-, :*)
    @eval Base.$op(a::X, b::X) where {X<:Field} = X(fieldtransform(a), broadcast($op, fielddata(a),fielddata(b)))
end

Base.:\(a::X, b::X) where {X<:Field} = X(fieldtransform(a), nan2zero.(fielddata(b) ./ fielddata(a)))

#  op(f::F, g::F) for different types F<:Field
for op in (:+, :-)
    @eval Base.$op(a::Field, b::Field) = $op(promote(a,b)...)
end

# fields and UniformScaling
LinearAlgebra.:*(f::X, J::UniformScaling) where X<:Field = J.λ * f
LinearAlgebra.:*(J::UniformScaling, f::X) where X<:Field = J.λ * f
LinearAlgebra.:\(J::UniformScaling, f::X) where X<:Field = (1/J.λ) * f

