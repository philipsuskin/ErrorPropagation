using MonteCarloMeasurements
include("../utils/coordinateConvert.jl")

using GeometryBasics
using GLMakie

using LinearAlgebra, Random

PARTICLE_COUNT = Int(1e5)
σ = 2
σᵣ = 1.1e-3 / 0.045 / sqrt(3)
R = 1

# ps = [[1, 0, 45]]
ps = [[1, 0, 45], [1, 90, 180], [1, 90, 270], [1, 45, 225]]
ps = map(p -> p .* [R, pi / 180, pi / 180], ps)

function plotSphere()
  ps_mc = map(p -> p +
    [
      # 0,
      Particles(PARTICLE_COUNT, MonteCarloMeasurements.Normal(0, σᵣ)),
      deg2rad(Particles(PARTICLE_COUNT, MonteCarloMeasurements.Normal(0, σ))),
      deg2rad(Particles(PARTICLE_COUNT, MonteCarloMeasurements.Normal(0, σ))) / sin(p[2] == 0 ? 1e-9 : p[2])
    ],
    ps
  )
  rs_mc = map(spherical_to_cartesian, ps_mc)

  fig = Figure(resolution = (900, 900), fontsize = 10)
  ax = Axis3(fig[1, 1], title = "Random Angle Simulation", xlabel = "X", ylabel = "Y", zlabel = "Z")

  for r_mc in rs_mc
      scatter!(ax, r_mc[1].particles, r_mc[2].particles, r_mc[3].particles, markersize = 2, color = :blue, label = "Particles")
  end

  sphere = Sphere(Point3f(0), 1)
  # sphere = GeometryBasics.mesh(Tesselation(sphere, 256))
  GLMakie.mesh!(ax, sphere, color = (:white, 0.25), transparency = true)

  # ax.limits = Tuple(map(extrema, map(o -> o.particles, r_mc)))
  ax.aspect = :data

  fig
end

function random_points_on_sphere_around(p)
    r = spherical_to_cartesian(p)
    t1 = normalize(nullspace(r')[:, 1])
    t2 = normalize(cross(r, t1))
    r_mc = [Particles(PARTICLE_COUNT) for _ in 1:3]
    for i in 1:PARTICLE_COUNT
        a, b = randn(2) .* (deg2rad(σ * R))
        v = a * t1 + b * t2
        θ = norm(v)
        point = θ ≈ 0 ? p : cos(θ) * p + sin(θ) * v / θ
        for j in 1:3
          r_mc[j].particles[i] = point[j]
        end
    end
    return map(cartesian_to_spherical, r_mc)
end

plotSphere()