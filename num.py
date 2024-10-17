import jax
import jax.numpy as jnp

def _erfcx_naive(x):
  """Compute erfcx using a Chebyshev expansion."""
  # The implementation is based on
  # [1] M. Shepherd and J. Laframboise,
  #     Chebyshev approximation of (1 + 2 * x) * exp(x**2) * erfc(x)
  #     https://www.ams.org/journals/mcom/1981-36-153/S0025-5718-1981-0595058-X/

  x_abs = jnp.abs(x)
  # TODO(b/180390310): The approximation quality can be made better by sweeping
  # the shift parameter '3.75'.
  y = (x_abs - 3.75) / (x_abs + 3.75)

  # The list of coefficients is taken from [1].
  coeff = [
      3e-21,
      9.7e-20,
      2.7e-20,
      -2.187e-18,
      -2.237e-18,
      5.0681e-17,
      7.4182e-17,
      -1.250795e-15,
      -1.864563e-15,
      3.33478119e-14,
      3.2525481e-14,
      -9.65469675e-13,
      1.94558685e-13,
      2.8687950109e-11,
      -6.3180883409e-11,
      -7.75440020883e-10,
      4.521959811218e-09,
      1.0764999465671e-08,
      -2.18864010492344e-07,
      7.74038306619849e-07,
      4.139027986073010e-06,
      -6.9169733025012064e-05,
      4.90775836525808632e-04,
      -2.413163540417608191e-03,
      9.074997670705265094e-03,
      -2.6658668435305752277e-02,
      5.9209939998191890498e-02,
      -8.4249133366517915584e-02,
      -4.590054580646477331e-03,
      1.177578934567401754080,
  ]

  result = -4e-21
  previous_result = 0.
  for i in range(len(coeff) - 1):
    result, previous_result = (
        2 * y * result - previous_result + coeff[i], result)
  result = y * result - previous_result + coeff[len(coeff) - 1]

  result = result / (1. + 2. * x_abs)

  # The approximation is only valid for positive x, so flip the integral.
  # TODO(b/180390310): Improve this approximation for negative values.
  result = jnp.where(
      x < 0., 2. * jnp.exp(jnp.square(x)) - result, result)
  result = jnp.where(x == float('inf'), 1.0, result)
  return result


def _erfcx_fwd(x):
  """Compute output, aux (collaborates with _erfcx_bwd)."""
  output = _erfcx_naive(x)
  return output, (x,)


def _erfcx_bwd(aux, g):
  x, = aux
  y = _erfcx_custom_gradient(x)
  px = 2. * x * y - (2. / jnp.sqrt(jnp.pi))
  return [px * g]


# def _erfcx_jvp(primals, tangents):
#   """Computes JVP for erfcx (supports JAX custom derivative)."""
#   x, = primals
#   dx, = tangents

#   y = _erfcx_custom_gradient(x)
#   numpy_dtype = dtype_util.as_numpy_dtype(
#       dtype_util.common_dtype([x], tf.float32))
#   px = 2. * x * y - numpy_dtype(2. / np.sqrt(np.pi))
#   return y, px * dx


@jax.custom_vjp
def _erfcx_custom_gradient(x):
  """Computes Erfcx(x) with correct custom gradient."""
  return _erfcx_naive(x)

_erfcx_custom_gradient.defvjp(_erfcx_fwd, _erfcx_bwd)

def erfcx(x, name=None):
  """Computes the scaled complementary error function exp(x**) * erfc(x).

  # References
  [1] M. Shepherd and J. Laframboise,
      Chebyshev approximation of (1 + 2 * x) * exp(x**2) * erfc(x)
      https://www.ams.org/journals/mcom/1981-36-153/S0025-5718-1981-0595058-X/

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    erfcx: erfcx(x) evaluated at `x`. A Tensor with the same shape and same
      dtype as `x`.
  """
  return _erfcx_custom_gradient(x)