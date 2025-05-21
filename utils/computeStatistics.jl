using QuadGK  # Numeric integration

function meanError(f, R)
    V = (4 / 3) * π * R^3

    integral = quadgk(r -> quadgk(θ -> quadgk(φ -> begin
        x = r * sin(θ) * cos(φ)
        y = r * sin(θ) * sin(φ)
        z = r * cos(θ)
        f(x, y, z) * r^2 * sin(θ)  # Integrand in spherical coordinates
    end, 0, 2π)[1], 0, π)[1], 0, R)[1]

    return integral / V
end

function meanErrorMonteCarlo(f, R, num_samples=10^6)
  V = (4 / 3) * π * R^3

  total = 0.0
  for _ in 1:num_samples
      r = R * cbrt(rand())  # Random radius
      θ = acos(1 - 2 * rand())  # Random polar angle
      φ = 2π * rand()  # Random azimuthal angle

      # Calculate Cartesian coordinates
      x = r * sin(θ) * cos(φ)
      y = r * sin(θ) * sin(φ)
      z = r * cos(θ)

      total += f(x, y, z)
  end

  return total / num_samples
end