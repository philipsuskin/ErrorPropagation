using TypedPolynomials
using SphericalHarmonicExpansions

using MPISphericalHarmonics, MPIUI

TypedPolynomials.@polyvar x y z

varianceCoeff = 1

L = 6

cx = SphericalHarmonicExpansions.SphericalHarmonicCoefficients(L, 0.045, true)
cy = SphericalHarmonicExpansions.SphericalHarmonicCoefficients(L, 0.045, true)
cz = SphericalHarmonicExpansions.SphericalHarmonicCoefficients(L, 0.045, true)

for l in 0:L
  for m in -l:l
    for cs in [cx, cy, cz]
      cs[l, m] = varianceCoeff
    end
  end
end

coeffs = MagneticFieldCoefficients([cx; cy; cz;;], 0.045)
# coeffs = MagneticFieldCoefficients("coeffs.h5")
# coeffs.coeffs[1].c .= 0.0

MagneticFieldViewer(coeffs)