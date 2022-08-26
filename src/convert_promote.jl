# convert and promote
# ============================================================

#  convert and promote
function Base.promote_rule(::Type{MF}, ::Type{FF}) where {F,Tf,Ti,d, MF<:MapField{F,Tf,Ti,d}, FF<:FourierField{F,Tf,Ti,d}} 
    return MF
end

function Base.convert(::Type{MF}, f::FourierField{F,Tf,Ti,d}) where {F,Tf,Ti,d, MF<:MapField{F,Tf,Ti,d}} 
    ftran = fieldtransform(f)
    return MF(ftran, plan(ftran) \ fielddata(f))
end

function Base.convert(::Type{FF}, f::MapField{F,Tf,Ti,d}) where {F,Tf,Ti,d, FF<:FourierField{F,Tf,Ti,d}} 
    ftran = fieldtransform(f)
    return FF(ftran, plan(ftran) * fielddata(f))
end

Base.convert(::Type{X}, f::X) where X<:Field = f::X

# Things that fall back to convert
# ============================================================

# convert to the corresponding dual field using FourierField or MapField
FourierField(f::X) where {X<:Field} = convert(FourierField(X),f)
MapField(f::X)     where {X<:Field} = convert(MapField(X),f)

# convert to the corresponding dual field using the explicit type
(::Type{MF})(f::Field{F,Tf,Ti,d}) where {F,Tf,Ti,d, MF<:MapField{F,Tf,Ti,d}} = convert(MF, f)
(::Type{FF})(f::Field{F,Tf,Ti,d}) where {F,Tf,Ti,d, FF<:FourierField{F,Tf,Ti,d}} = convert(FF, f)
