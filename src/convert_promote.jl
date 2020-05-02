# convert and promote
# ============================================================

#TODO: Try and make this generic with FieldMap and FieldFourier
## !!! here we use the convention that transform first, data second

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
function Base.promote_rule(::Type{Xmap{F,Tf,Ti,d}}, ::Type{Xfourier{F,Tf,Ti,d}}) where {Tf,d,F<:Transform{Tf,d},Ti} 
    return Xmap{F,Tf,Ti,d}
end

function Base.convert(::Type{Xmap{F,Tf,Ti,d}}, f::Xfourier{F,Tf,Ti,d}) where {Tf,d,F<:Transform{Tf,d},Ti} 
    return Xmap(f.ft, plan(f.ft) \ f.f)
end

function Base.convert(::Type{Xfourier{F,Tf,Ti,d}}, f::Xmap{F,Tf,Ti,d}) where {Tf,d,F<:Transform{Tf,d},Ti} 
    return Xfourier(f.ft, plan(f.ft) * f.f)
end

Base.convert(::Type{X}, f::X) where X<:Field = f::X

# make constructors fall back to convert
Xfourier(f::Xfield{F,Tf,Ti,d}) where {Tf,d,F<:Transform{Tf,d},Ti} = convert(Xfourier{F,Tf,Ti,d}, f)
Xmap(f::Xfield{F,Tf,Ti,d})     where {Tf,d,F<:Transform{Tf,d},Ti} = convert(Xmap{F,Tf,Ti,d}, f)

# Use FT to convert between Xmap vrs Xfourier
Base.:*(::Type{F}, f::Xfourier{F,Tf,Ti,d})  where {Tf,d,F<:Transform{Tf,d},Ti} = f
Base.:*(::Type{F}, f::Xmap{F,Tf,Ti,d})      where {Tf,d,F<:Transform{Tf,d},Ti} = convert(Xfourier{F,Tf,Ti,d}, f)
Base.:\(::Type{F}, f::Xfourier{F,Tf,Ti,d})  where {Tf,d,F<:Transform{Tf,d},Ti} = convert(Xmap{F,Tf,Ti,d}, f)
Base.:\(::Type{F}, f::Xmap{F,Tf,Ti,d})      where {Tf,d,F<:Transform{Tf,d},Ti} = f
