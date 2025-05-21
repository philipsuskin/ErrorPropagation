using TypedPolynomials
using SphericalHarmonicExpansions

using MPIFiles
using MonteCarloMeasurements

# Variances
σ²ᵣ = [0.1, 0.1, 0.1]  # [σ²ₓ, σ²ᵧ, σ²𝓏]

# t-design
tDes = loadTDesign(12, 86, 0.045)

TypedPolynomials.@polyvar x y z

l = 5
m = 2
Zₗᵐ = SphericalHarmonicExpansions.zlm(l, m, x, y, z)

println("Zₗᵐ: $Zₗᵐ")

# Error Propagation Formula
function getVarianceFirstOrder(σ²ᵣ, Zₗᵐ, Cov = zeros(3, 3))
  σ² = zero(Zₗᵐ)

  # Derivatives
  δZₗᵐ = [differentiate(Zₗᵐ, variable) for variable in (x, y, z)]

  # Variance contribution
  σ² += sum(δZₗᵐ[i]^2 * σ²ᵣ[i] for i in 1:3)

  # Covariance contribution
  σ² += 2 * (δZₗᵐ[1]*δZₗᵐ[2]*Cov[1, 2] +
             δZₗᵐ[1]*δZₗᵐ[3]*Cov[1, 3] +
             δZₗᵐ[2]*δZₗᵐ[3]*Cov[2, 3])

  return σ²
end
function getVariance(σ²ᵣ, Zₗᵐ, Cov = zeros(3, 3))
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

  Zₗᵐ² = Zₗᵐ * Zₗᵐ
  println("Zₗᵐ²: $Zₗᵐ²")
  for term in Zₗᵐ².terms
    coeff = term.coefficient
    monomial = term.monomial
    for (i, variable) in enumerate((x, y, z))
      exponent = monomial.exponents[i]
      variance = σ²ᵣ[i]
      coeff *= E(exponent, variance)
    end

    Eᵪ₂ += coeff
  end
  println("Eᵪ₂: $Eᵪ₂")

  for term in Zₗᵐ.terms
    coeff = term.coefficient
    monomial = term.monomial
    for (i, variable) in enumerate((x, y, z))
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

σ² = getVariance(σ²ᵣ, Zₗᵐ)
println("σ²: $σ²")

N = 10^6
ϵ = [Particles(N, Normal(0, σ²ᵣ[1])),
     Particles(N, Normal(0, σ²ᵣ[2])),
     Particles(N, Normal(0, σ²ᵣ[3]))]

f = (rx, ry, rz) -> Zₗᵐ(x => rx, y => ry, z => rz)
for pos in eachcol(tDes.positions)
  rx, ry, rz = 0, 0, 0
  fMC = f(rx + ϵ[1], ry + ϵ[2], rz + ϵ[3])
  varMC = var(fMC.particles)
  println("Variance from Monte Carlo: $varMC")
  exit()

  varFormula = σ²(x => rx, y => ry, z => rz)
  println("Variance from formula: $varFormula")
end