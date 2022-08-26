# XFields

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EthanAnderes.github.io/XFields.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EthanAnderes.github.io/XFields.jl/dev)
[![Build Status](https://travis-ci.com/EthanAnderes/XFields.jl.svg?branch=master)](https://travis-ci.com/EthanAnderes/XFields.jl)



## Under construction ... 

Installation

```julia
julia> using Pkg
julia> pkg"add https://github.com/EthanAnderes/XFields.jl"
```


## Pre-defined concrete fields Xmap and Xfourier 


```julia
Xmap{Tm, Ti, To ,d}     <: Xfield # concrete map field type  
Xfourier{Tm, Ti, To ,d} <: Xfield # concrete fourier field type   
```
`Tm` is a Transform type which holds enough information to transform a Xmap array to a Xfourier array and back.

`Ti` is the map pixel eltype; `To` is the fourier pixel eltype. 
`d` is the dimension of both. These types must match the following required methods for the transform `tm<:Transform`

```julia
szi = size_in(tm) 
szo = size_out(tm) 
Ti  = eltype_in(tm) 
To  = eltype_out(tm) 
pft = plan(tm)

```


## Generic Field


```julia 
Field{Tm<:Transform, Ti<:Number, To<:Number, d}   # abstract type
FourierField{Tm, Ti, To ,d} <: Field              # abstract type
MapField{Tm, Ti, To ,d}     <: Field              # abstract type
```

FourierField and MapField are used for mapping concrete types to duals. Used when defining a new concrete Field type.

```julia
FourierField(::Type{X}) # -> concrete field type X to dual type
MapField(::Type{X})     # -> concrete field type X to dual type
```

```julia
# for f::Field 

FourierField(f)   -> f′::FourierField, convert to dual field
MapField(f)       -> f′::MapField,     convert to dual field

fieldtransform(f) -> tm::Transform
fielddata(f)      -> storage_data::Array

f[!]              -> fielddata(FourierField(f))
f[:]              -> fielddata(MapField(f))

tm * f            -> FourierField(f)
tm \ f            -> MapField(f)
```

```julia
AbstractLinearOp                      # abstract
DiagOp{X<:Field} <: AbstractLinearOp  # concrete

# base methods

diag(O::DiagOp)
inv(O::AbstractLinearOp) 
adjoint(O::AbstractLinearOp) 
*(O::AbstractLinearOp, f::Field) 
\(O::AbstractLinearOp, f::Field)
```



# Quickstart: Pixel field with fourier operator 

Here is a quick example of a one dimensional field period on the interval [0, 1). We will construct two operators A, B where A represents the derivative operator, diagonal in Fourier space, and B represents a pixel masking operator, diagonal in pixel space.

First we defined the transform between pixel fields and Fourier fields using the package FFTransforms which provides a template for the transform types which parameterize Xfields.

```julia
using XFields
using FFTransforms

npix   = 128
period = 1.0 
Wt = r𝕎(npix, period)
```

The transform `Wt` represents regular real discrete Fourier transform. We can scale it to obtain a discrete version of the integral transform (i.e. with pre-factor Δpix / (2π)^(d/2) with d = 1 in this example). 

```julia
scale = ordinary_scale(Wt) # 0.003116736565636193
Ft = scale * Wt
```

The transform `Ft` holds enough information to be able to generate a concrete FFT plan, extracted via the method `plan`

```julia
FFt = plan(Ft)
fx = randn(npix)
fk = FFt * fx
sum(abs2, fx .- FFt \ fk)
``` 

The transform `Ft` can also be used to generate a basis agnostic field type. 

```julia
f = Xmap(Ft, fx)
```

The constructor `Xmap` generates a field which is stored in pixel coordinates and also holds `Ft` so it can automatically convert to Fourier basis and back to pixel basis when needed. To see this in action lets now defined `A` and `B` as described above 

```julia
ik = im * freq(Ft)[1]  # diagonal elements of d/dx in Fourier space
mx = rand(npix) .< 0.5 # pixel mask
A = Xfourier(Ft, ik) |> DiagOp
B = Xmap(Ft, mx) |> DiagOp
```

Now we can use `A` and `B` as matrix operators on `f` and, internally, the transforms to and from Fourier space are handled automatically 

```julia
af = A * f
bf = B * f
cf = A * B * f  - B * A * f
```

Each on of `af`, `bf` and `cf` is another instance of an `Xmap{typeof(Ft)}` whos pixel values can be extracted with `fielddata` or with `:` indexing 

```julia
julia> af[:]
128-element Array{Float64,1}:
  110.58746393123022
  -57.2551301738924
 -252.01103182546902
  319.13605401234
    ⋮

julia> bf[:]
128-element Array{Float64,1}:
 -0.9205754019342572
  0.4769001923181931
 -0.0
 -1.0349069395521513
    ⋮


 julia> cf[:]
128-element Array{Float64,1}:
 -110.2082261043798
  208.17650662242562
 -213.571475779447
 -285.53106590672445
  114.28788227443992
    ⋮

``` 

Also the Fourier coefficients can be extracted with `!` indexing



```julia
julia> af[!]
65-element Array{Complex{Float64},1}:
  2.1259915449510525e-15 + 0.0im
     0.09809052721197642 - 0.11935338002108792im
      0.7697055591088555 - 0.22907176932738127im
     -0.7313295036626855 - 0.6181242403839325im
    -0.11815322659743498 + 0.2982645102415518im
    ⋮

julia> bf[!]
65-element Array{Complex{Float64},1}:
   0.043522929531874074 + 0.0im
  -0.007868454304947346 + 0.0021877926624518994im
  -0.016332493138807334 - 0.02188401856709664im
  -0.010917750036908718 + 0.01704245672465362im
    ⋮

julia> cf[!]
65-element Array{Complex{Float64},1}:
    5.681171544522148 + 0.0im
   0.6721949791035988 + 2.6712627645708547im
   -5.515015479213712 - 1.0806708378945662im
   -4.783820311291526 + 0.4231963572336103im
    ⋮
```

We also are able to perform basic operations on these fields via 

```
2 * A * f -  √B \ af
```
where mixes of Xfourier or Xmap types are appropriately and automatically converged to a common basis before the operation is performed.



## Quickstart: Fourier field with pixel operator






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

LinearAlgebra.dot(f::Xfield{F}, g::Xfield{F}) where F = dot(f[:], g[:]) .* Ωpix(F)
```
where `Ωpix(F)` is a method defined for types `F<:𝔽{Tf,d,opt}`






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


