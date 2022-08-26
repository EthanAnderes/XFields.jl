# getindex for :, ! and *, / with the transform 
# ============================================================

#  f[!] -> fourier data storage array, f[:] -> map storage array
Base.getindex(f::Field, ::typeof(!)) = fielddata(FourierField(f))
Base.getindex(f::Field, ::Colon)     = fielddata(MapField(f))

# f[()] takes a storage array and unpackes it as a tuple. 
# Base.getindex(f::Field, ::Tuple{})   = (fielddata(MapField(f)),)
# We will reserve this for custom usage ...
# it was intended for processing the data from a field to be lensed but
# instead I decided to use f[L::AbstractFlow] for this


# convert to the corresponding dual field using the transform itself
Base.:*(ft::F, f::Field{F,Tf,Ti,d})  where {F,Tf,Ti,d} = FourierField(f)
Base.:\(ft::F, f::Field{F,Tf,Ti,d})  where {F,Tf,Ti,d} = MapField(f)


# field operations
# ============================================================

# op(f::Field, n::Number) and op(n::Number, f::Field)
for op in (:+, :-, :*, :/, :\)
    @eval Base.$op(f::X, n::Number) where X<:Field = X(fieldtransform(f), broadcast($op, fielddata(f),n))
    @eval Base.$op(n::Number, f::X) where X<:Field = X(fieldtransform(f), broadcast($op, n, fielddata(f)))
end

Base.:^(f::X, n::Number)  where X<:Field = X(fieldtransform(f), broadcast(^, fielddata(f),n))
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

Base.:\(a::X, b::X) where {X<:Field} = X(fieldtransform(a), nan2zero.(fielddata(a) .\ fielddata(b)))
Base.:/(a::X, b::X) where {X<:Field} = X(fieldtransform(a), nan2zero.(fielddata(a) ./ fielddata(b)))

#  op(f::F, g::F) for different types F<:Field
for op in (:+, :-)
    @eval Base.$op(a::Field, b::Field) = $op(promote(a,b)...)
end

# fields and UniformScaling
LinearAlgebra.:*(f::X, J::UniformScaling) where X<:Field = J.λ * f
LinearAlgebra.:*(J::UniformScaling, f::X) where X<:Field = J.λ * f
LinearAlgebra.:\(J::UniformScaling, f::X) where X<:Field = (1/J.λ) * f

