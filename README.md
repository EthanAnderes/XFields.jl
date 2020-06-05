# XFields

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EthanAnderes.github.io/XFields.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EthanAnderes.github.io/XFields.jl/dev)
[![Build Status](https://travis-ci.com/EthanAnderes/XFields.jl.svg?branch=master)](https://travis-ci.com/EthanAnderes/XFields.jl)



## Under construction ... 

Installation

```
julia> using Pkg
julia> pkg"add https://github.com/EthanAnderes/XFields.jl#master"
```


## Transforms

XFields is intended to be used when working with pairs of multidimensional arrays `ai::Array{Tf,d}` and `ao::Array{Ti,d}` where are connected via a linear transformation. 


Generally the main way to utilize `XFields` is to define a transform type 
`𝔽{Tf,d,opt} <: Transform{Tf,d}` which encodes enough information 
to instantiate the linear transformation from `ai` to `ao` and it's inverse. 
This is done via the method `plan(ft::𝔽{Tf,d,opt})` which returns another object that left-multiplies `ai` and left-divides `ao`. In particular one needs to define

- `plan(ft)` which produces an object that explicitly represents the transform
- `plan(ft) * ai -> ao` 
- `plan(ft) \ ao -> ai`

In addition the following methods need to be defined

- `size_in(ft::𝔽{Tf,d,opt}) -> size(ai)`
- `size_out(ft::𝔽{Tf,d,opt}) -> size(ao)`
- `eltype_in(ft:𝔽{Tf,d,opt}) -> eltype(ai)` 
- `eltype_out(ft:𝔽{Tf,d,opt}) -> eltype(ao)`


The above transform `𝔽{Tf,d,opt} <: Transform{Tf,d}` can now be used with the preloaded array wrapper `fi::Xmap{F<:𝔽{Tf,d,opt}}` and `fo::Xfourier{F<:𝔽{Tf,d,opt}}` using the new transform object `F::𝔽{Tf,d,opt}` which automatically utalizes `F` to convert

```julia
ai = rand(Tf,d)
ao = F * ai

# instanciation
fi = Xmap(F,ai)
fo = Xfourier(F,ao)

# implicit convert fi <-> fo
fi == Xmap(fo)
fo == Xfourier(fi)

# explicit convert fi <-> fo
fi == F * fo
fi == F * fi
fo == F \ fi
fo == F \ fo
```


Now one can extend operators such as 

```julia
function Base.getindex(f::Xfield, sym::Symbol)
    (sym == :k) ? Xfourier(f).f :
    (sym == :l) ? Xfourier(f).f :
    (sym == :x) ? Xmap(f).f :
    error("index is not defined")
end

LinearAlgebra.dot(f::Xfield{F}, g::Xfield{F}) where F = dot(f[:], g[:]) .* Ωx(F)
```
where `Ωx(F)` is a method defined for types `F<:𝔽{Tf,d,opt}`






## Fields

One may also define their own field type, say `Ymap{Tf,d,...}` and `Yfourier{Tf,d,...}`

`abstract type YField{F<:Transform,Tf,Ti,d} <: Field{F,Tf,Ti,d}`

```julia
struct Ymap{F<:Transform, Tf<:Number, Ti<:Number, d} <: MapField{F,Tf,Ti,d}
    ft::F
    f::Array{Tf,d}
    function Ymap{F,Tf,Ti,d}(ft::F, f::Array{Tf,d})  where {Tf,Ti,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_in(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
end

struct Yfourier{F<:Transform, Tf<:Number, Ti<:Number, d}  <: FourierField{F,Tf,Ti,d}
    ft::F
    f::Array{Ti,d}
    function Yfourier{F,Tf,Ti,d}(ft::F, f::Array{Ti,d})  where {Tf,Ti,d,F<:Transform{Tf,d}}
        @assert Ti == eltype_out(ft)
        @assert size(f) == size_out(ft)
        new{F,Tf,Ti,d}(ft, f)
    end
end

```


