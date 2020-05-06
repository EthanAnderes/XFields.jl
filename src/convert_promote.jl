# convert and promote
# ============================================================

@generated function fieldtransform(x::Field)
    ft = fieldname(x, 1)
    return quote
        $(Expr(:meta, :inline))
        x.$ft
    end
end

@generated function fielddata(x::Field)
    f = fieldname(x, 2)
    return quote
    	$(Expr(:meta, :inline))
    	x.$f
    end
end

#  convert and promote
function Base.promote_rule(::Type{MF}, ::Type{FF}) where {Tf,Ti,d, F<:Transform{Tf,d}, MF<:MapField{F,Tf,Ti,d}, FF<:FourierField{F,Tf,Ti,d}} 
    return MF
end

function Base.convert(::Type{MF}, f::FourierField{F,Tf,Ti,d}) where {Tf,Ti,d, F<:Transform{Tf,d}, MF<:MapField{F,Tf,Ti,d}} 
    return MF(f.ft, plan(f.ft) \ f.f)
end

function Base.convert(::Type{FF}, f::MapField{F,Tf,Ti,d}) where {Tf,Ti,d, F<:Transform{Tf,d}, FF<:FourierField{F,Tf,Ti,d}} 
    return FF(f.ft, plan(f.ft) * f.f)
end

Base.convert(::Type{X}, f::X) where X<:Field = f::X
