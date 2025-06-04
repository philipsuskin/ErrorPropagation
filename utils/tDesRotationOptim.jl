using Manifolds, ManifoldDiff
using LinearAlgebra
using Manopt

# Define cost funtion ∑ᵢ d(p*tDesᵢ,estimatedᵢ)², where d is the geodesic distance on the sphere S², tDesᵢ is a point from the tdesign positions in the CAD model, estimatedᵢ is a point from sensor estimation by their rotation matrices and p is a rotation matrix
function findtDesRotation(tDesign, estimatedPos)
  tDes = copy(tDesign.positions)
  S2 = Manifolds.Sphere(2) # Unit Sphere
  SO3 = SpecialOrthogonal(3) # Rotation group

  G(p) = 1 / (2 * N) * mapreduce(i -> distance(S2, p * tDes[:, i], estimatedPos[:, 1])^2, +, 1:N)
  # Cost funtion for Manopt.jl
  g(M, p) = G(p)

  # Define gradient using automatic diffentiation https://manoptjl.org/stable/tutorials/AutomaticDifferentiation/
  rb_onb_fwdd = ManifoldDiff.TangentDiffBackend(ManifoldDiff.AutoForwardDiff())
  grad_g(M, p) = Manifolds.gradient(M, G, p, rb_onb_fwdd)
  #@info "" check_gradient(SO3, g, grad_g)

  # Initial value
  R0 = project(SO3, I(3) .* 1.0) #rand(SO3)

  # Gradient Descent https://redoblue.github.io/techblog/2019/02/22/optimization-riemannian-manifolds/
  Rest = gradient_descent(SO3, g, grad_g, R0;
    # debug=[:Iteration, (:Change, "|Δp|: %1.9f |"), (:Cost, " F(x): %1.11f | "), "\n", :Stop, 5],
    stopping_criterion=StopWhenGradientNormLess(1e-6) | StopAfterIteration(100)
  )

  return Rest * tDes
end