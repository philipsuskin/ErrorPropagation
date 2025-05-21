using TypedPolynomials
using SphericalHarmonicExpansions
using MonteCarloMeasurements

function getVariance(σ²ᵣ, χ, Cov = zeros(3, 3))
  function doublefactorial(n::Integer)
    fact = one(n)
    for m in iseven(n)+1:2:n
        fact *= m
    end
    return fact
  end

  function E(exponent::Integer, variance)
    if exponent < 0
      throw(ArgumentError("Exponent must be a non-negative integer."))
    end

    if isodd(exponent)
      return 0.0
    else iseven(exponent)
      return doublefactorial(exponent - 1) * variance^exponent
    end
  end

  σ² = 0.0
  Eᵪ₂ = 0.0
  Eᵪ = 0.0

  χ² = χ * χ
  for term in χ².terms
    coeff = term.coefficient
    monomial = term.monomial
    for i in eachindex(monomial.exponents)
      exponent = monomial.exponents[i]
      variance = σ²ᵣ[i]
      coeff *= E(exponent, variance)
    end

    Eᵪ₂ += coeff
  end
  println("Eᵪ₂: $Eᵪ₂")

  for term in χ.terms
    coeff = term.coefficient
    monomial = term.monomial
    for i in eachindex(monomial.exponents)
      exponent = monomial.exponents[i]
      variance = σ²ᵣ[i]
      coeff *= E(exponent, variance)
    end

    Eᵪ += coeff
  end
  Eᵪ² = Eᵪ * Eᵪ
  println("Eᵪ²: $Eᵪ²")

  σ² = Eᵪ₂ - Eᵪ²
  return σ²
end

function getVarianceMonteCarlo(σ²ᵣ, χ, particleCount = 10^4; isLocal = false)
  ϵ = [Particles(particleCount, Normal(0, σ²ᵣ[1])),
       Particles(particleCount, Normal(0, σ²ᵣ[2])),
       Particles(particleCount, Normal(0, σ²ᵣ[3]))]

  f = (rx, ry, rz) -> χ(x => rx + ϵ[1], y => ry + ϵ[2], z => rz + ϵ[3])

  return isLocal ? f : var(f(0, 0, 0).particles)
end