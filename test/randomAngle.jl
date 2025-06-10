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

  fig = GLMakie.Figure(resolution = (900, 900), fontsize = 10)
  ax = Axis3(fig[1, 1], title = "Random Angle Simulation", xlabel = "X", ylabel = "Y", zlabel = "Z")

  for r_mc in rs_mc
      scatter!(ax, r_mc[1].particles, r_mc[2].particles, r_mc[3].particles, markersize = 2, color = :blue, label = "Particles")
  end

  sphere = GLMakie.Sphere(Point3f(0), 1)
  # sphere = GeometryBasics.mesh(Tesselation(sphere, 256))
  GLMakie.mesh!(ax, sphere, color = (:white, 0.25), transparency = true)

  # ax.limits = Tuple(map(extrema, map(o -> o.particles, r_mc)))
  ax.aspect = :data

  fig
end

function plotRandomDirection()
  pₑ_mc = [Particles{Float64, PARTICLE_COUNT}(1), acos(Particles(PARTICLE_COUNT, Uniform(-1, 1))), Particles(PARTICLE_COUNT, Uniform(0, 2π))]

  rₑ_mc = spherical_to_cartesian(pₑ_mc)

  fig = GLMakie.Figure(resolution = (900, 900), fontsize = 10)
  ax = Axis3(fig[1, 1], title = "Random Direction Simulation", xlabel = "X", ylabel = "Y", zlabel = "Z")
  scatter!(ax, rₑ_mc[1].particles, rₑ_mc[2].particles, rₑ_mc[3].particles, markersize = 2, color = :blue, label = "Particles")
  sphere = GLMakie.Sphere(Point3f(0), 1)
  GLMakie.mesh!(ax, sphere, color = (:white, 0.25), transparency = true)
  ax.limits = Tuple(map(extrema, map(o -> o.particles, rₑ_mc)))
  ax.aspect = :data
  fig
end

# plotSphere()
plotRandomDirection()