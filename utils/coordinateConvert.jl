function cartesian_to_spherical(v)
  x, y, z = v[1], v[2], v[3]

  r = sqrt(x^2 + y^2 + z^2)
  θ = acos(z / r)
  ϕ = atan(y, x)
  return [r, θ, ϕ]
end

function spherical_to_cartesian(v)
  r, θ, ϕ = v[1], v[2], v[3]

  x = r * sin(θ) * cos(ϕ)
  y = r * sin(θ) * sin(ϕ)
  z = r * cos(θ)
  return [x, y, z]
end