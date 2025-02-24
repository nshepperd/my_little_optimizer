import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable
import pytest

jax.config.update("jax_enable_x64", True)

from my_little_optimizer.gaussian_process import GP


def rbf_kernel(length_scale: float = 1.0) -> Callable:
    """Simple RBF kernel for testing"""
    def kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
        return jnp.exp(-jnp.sum((x1 - x2)**2) / (2 * length_scale**2))
    return kernel

def make_test_data(n_points: int = 10, dim: int = 1, seed: int = 42) -> Tuple[jax.Array, jax.Array]:
    """Generate well-conditioned test data"""
    key = jax.random.PRNGKey(seed)
    xs = jax.random.uniform(key, (n_points, dim))
    ys = jnp.sin(2 * jnp.pi * xs.sum(-1))  # Simple periodic function
    return xs, ys

class TestGPNumericalStability:
    """Test suite for GP numerical stability"""
    
    def setup_method(self):
        """Setup baseline GP with well-conditioned data"""
        self.kernel = rbf_kernel()
        self.xs_base, self.ys_base = make_test_data()
        
        # Create both float32 and float64 versions
        self.gp32 = GP(self.kernel, jnp.array(0.1, dtype=jnp.float32))
        self.gp64 = GP(self.kernel, jnp.array(0.1, dtype=jnp.float64))

    def get_baseline_precision(self, xs, ys, x_test) -> float:
        """Get baseline precision for well-conditioned inputs"""
        mean32, var32 = self.gp32.predict(
            xs.astype(jnp.float32),
            ys.astype(jnp.float32),
            x_test.astype(jnp.float32)
        )
        mean64, var64 = self.gp64.predict(
            xs.astype(jnp.float64), 
            ys.astype(jnp.float64),
            x_test.astype(jnp.float64)
        )
        
        rel_diff_mean = jnp.abs((mean32 - mean64) / mean64.clip(1e-8))
        rel_diff_var = jnp.abs((var32 - var64) / var64.clip(1e-8))
        
        return max(float(rel_diff_mean), float(rel_diff_var))

    def test_baseline_precision(self):
        """Verify baseline precision with well-conditioned data"""
        x_test = jnp.array([0.5])
        precision = self.get_baseline_precision(self.xs_base, self.ys_base, x_test)
        assert precision < 1e-3, f"Baseline precision {precision} worse than expected"

    def test_close_points(self):
        """Test numerical stability with very close points"""
        eps = 1e-6
        xs = jnp.array([[0.0], [eps]])
        ys = jnp.array([0.0, 0.0])
        x_test = jnp.array([eps/2])
        
        baseline = self.get_baseline_precision(self.xs_base, self.ys_base, x_test)
        precision = self.get_baseline_precision(xs, ys, x_test)
        
        assert precision < baseline * 100, f"Close points precision {precision} much worse than baseline {baseline}"

    def test_distant_points(self):
        """Test numerical stability with very distant points"""
        xs = jnp.array([[0.0], [1e6]])
        ys = jnp.array([0.0, 0.0])
        x_test = jnp.array([0.5])
        
        baseline = self.get_baseline_precision(self.xs_base, self.ys_base, x_test)
        precision = self.get_baseline_precision(xs, ys, x_test)
        
        assert precision < baseline * 100, f"Distant points precision {precision} much worse than baseline {baseline}"

    def test_small_noise(self):
        """Test numerical stability with very small noise"""
        gp32_small = GP(self.kernel, jnp.array(1e-10, dtype=jnp.float32))
        gp64_small = GP(self.kernel, jnp.array(1e-10, dtype=jnp.float64))
        
        x_test = jnp.array([0.5])
        mean32, var32 = gp32_small.predict(
            self.xs_base.astype(jnp.float32),
            self.ys_base.astype(jnp.float32),
            x_test.astype(jnp.float32)
        )
        mean64, var64 = gp64_small.predict(
            self.xs_base.astype(jnp.float64),
            self.ys_base.astype(jnp.float64),
            x_test.astype(jnp.float64)
        )
        
        precision = max(
            float(jnp.abs((mean32 - mean64) / mean64.clip(1e-8))),
            float(jnp.abs((var32 - var64) / var64.clip(1e-8)))
        )
        
        assert precision < 0.01

    def test_gradients(self):
        """Test numerical stability of gradients"""
        def loss_fn(xs, ys, x_test):
            mean, var = self.gp32.predict(xs, ys, x_test)
            return mean
        
        grad_fn = jax.grad(loss_fn, argnums=0)
        
        x_test = jnp.array([0.5])
        grad32 = grad_fn(
            self.xs_base.astype(jnp.float32),
            self.ys_base.astype(jnp.float32),
            x_test.astype(jnp.float32)
        )
        
        grad64 = jax.grad(loss_fn, argnums=0)(
            self.xs_base.astype(jnp.float64),
            self.ys_base.astype(jnp.float64),
            x_test.astype(jnp.float64)
        )
        
        rel_diff = jnp.max(jnp.abs((grad32 - grad64) / grad64))
        assert rel_diff < 1e-2, f"Gradient precision {rel_diff} worse than expected"

class TestGPInvariances:
    """Test suite for GP invariance properties"""
    
    def setup_method(self):
        """Setup GP with standard parameters"""
        self.kernel = rbf_kernel()
        self.gp = GP(self.kernel, jnp.array(0.1))
        self.xs, self.ys = make_test_data()

    def test_permutation_invariance(self):
        """Test that predictions are invariant to permutation of training points"""
        x_test = jnp.array([0.5])
        
        # Original prediction
        mean1, var1 = self.gp.predict(self.xs, self.ys, x_test)
        
        # Permuted prediction
        perm = jax.random.permutation(jax.random.PRNGKey(0), len(self.xs))
        mean2, var2 = self.gp.predict(self.xs[perm], self.ys[perm], x_test)
        
        assert jnp.allclose(mean1, mean2, rtol=1e-5)
        assert jnp.allclose(var1, var2, rtol=1e-5)

    def test_translation_invariance(self):
        """Test that predictions transform correctly under translation"""
        x_test = jnp.array([0.5])
        offset = jnp.array([1.0])
        
        # Original prediction
        mean1, var1 = self.gp.predict(self.xs, self.ys, x_test)
        
        # Translated prediction
        mean2, var2 = self.gp.predict(
            self.xs + offset, 
            self.ys,
            x_test + offset
        )
        
        assert jnp.allclose(mean1, mean2, rtol=1e-5)
        assert jnp.allclose(var1, var2, rtol=1e-5)

class TestGPCorrectness:
    """Test suite for GP basic correctness"""
    
    def setup_method(self):
        """Setup GP with standard parameters"""
        self.kernel = rbf_kernel()
        self.gp = GP(self.kernel, jnp.array(0.0))  # No noise for exact interpolation
        self.xs, self.ys = make_test_data()

    def test_positive_variance(self):
        """Test that predicted variance is always positive"""
        x_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
        means, vars = self.gp.predictb(self.xs, self.ys, x_test)
        
        assert jnp.all(vars >= 0)

    def test_increasing_variance(self):
        """Test that variance increases away from training points"""
        x_test = jnp.linspace(-1, 2, 100).reshape(-1, 1)
        means, vars = self.gp.predictb(self.xs, self.ys, x_test)
        
        # Find distances to nearest training point
        ds = jnp.abs(x_test[:, None] - self.xs.squeeze(-1))

        dists = jnp.min(jnp.abs(x_test - self.xs.squeeze(-1)), axis=1)
        
        # Check correlation between distance and variance
        correlation = jnp.corrcoef(dists, vars)[0, 1]
        assert correlation > 0.5, "Variance should generally increase with distance from training points"

if __name__ == "__main__":
    pytest.main([__file__])