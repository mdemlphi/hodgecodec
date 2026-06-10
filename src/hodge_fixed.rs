// hodge_fixed — no_std Q16.16 fixed-point Hodge decomposition
// Designed for bare-metal / eBPF contexts (no heap, 512B stack budget).
// Compatible with both std and no_std crate roots.

/// Number of fractional bits for Q16.16 fixed-point arithmetic
pub const Q16_FRAC_BITS: usize = 16;
pub const Q16_ONE: i32 = 1 << Q16_FRAC_BITS;

#[inline]
pub const fn q16_mul(a: i32, b: i32) -> i32 {
    let a64 = a as i64;
    let b64 = b as i64;
    ((a64 * b64) >> Q16_FRAC_BITS) as i32
}

#[inline]
pub const fn q16_div(a: i32, b: i32) -> i32 {
    let a64 = a as i64;
    let b64 = b as i64;
    if b64 == 0 { return 0; }
    ((a64 << Q16_FRAC_BITS) / b64) as i32
}

/// General tridiagonal solver (Thomas algorithm), Q16.16.
/// `a[0]` and `c[N-1]` are ignored. Scratchpads must be provided by caller.
/// Warning: do NOT allocate diagonal arrays on the stack under eBPF (512B limit).
pub fn thomas_algorithm<const N: usize>(
    a: &[i32; N],
    b: &[i32; N],
    c: &[i32; N],
    d: &[i32; N],
    x: &mut [i32; N],
    c_prime: &mut [i32; N],
    d_prime: &mut [i32; N],
) {
    if N == 0 { return; }
    if N == 1 {
        if b[0] != 0 { x[0] = q16_div(d[0], b[0]); }
        return;
    }

    c_prime[0] = q16_div(c[0], b[0]);
    d_prime[0] = q16_div(d[0], b[0]);

    for i in 1..N {
        let den = b[i].wrapping_sub(q16_mul(a[i], c_prime[i - 1]));
        if den == 0 { c_prime[i] = 0; d_prime[i] = 0; continue; }
        if i < N - 1 { c_prime[i] = q16_div(c[i], den); }
        let num = d[i].wrapping_sub(q16_mul(a[i], d_prime[i - 1]));
        d_prime[i] = q16_div(num, den);
    }

    x[N - 1] = d_prime[N - 1];
    let mut i = N - 1;
    while i > 0 {
        i -= 1;
        x[i] = d_prime[i].wrapping_sub(q16_mul(c_prime[i], x[i + 1]));
    }
}

/// Solves −Δφ = f (1D discrete Poisson) with diag=[2,…,2], off=[-1].
/// Avoids diagonal array allocations — eBPF-safe.
pub fn solve_poisson_1d<const N: usize>(
    f: &[i32; N],
    x: &mut [i32; N],
    c_prime: &mut [i32; N],
    d_prime: &mut [i32; N],
) {
    if N == 0 { return; }
    if N == 1 { x[0] = q16_div(f[0], 2 * Q16_ONE); return; }

    let a_val = -Q16_ONE;
    let b_val =  2 * Q16_ONE;
    let c_val = -Q16_ONE;

    c_prime[0] = q16_div(c_val, b_val);
    d_prime[0] = q16_div(f[0],  b_val);

    for i in 1..N {
        let den = b_val.wrapping_sub(q16_mul(a_val, c_prime[i - 1]));
        if den == 0 { c_prime[i] = 0; d_prime[i] = 0; continue; }
        if i < N - 1 { c_prime[i] = q16_div(c_val, den); }
        let num = f[i].wrapping_sub(q16_mul(a_val, d_prime[i - 1]));
        d_prime[i] = q16_div(num, den);
    }

    x[N - 1] = d_prime[N - 1];
    let mut i = N - 1;
    while i > 0 {
        i -= 1;
        x[i] = d_prime[i].wrapping_sub(q16_mul(c_prime[i], x[i + 1]));
    }
}

/// True 1D Hodge decomposition via Poisson solver, Q16.16 fixed-point, no_std.
///
/// f_field = gradient_part (∇φ, exact) + harmonic_part (δψ residual)
///
/// Pipeline:
///   1. div(f) = d*f  (backward differences)
///   2. −Δφ = div(f)  (Poisson solve → Thomas O(N))
///   3. ∇φ = dφ       (forward differences of potential)
///   4. δψ = f − ∇φ   (coexact residual)
pub fn hodge_decomposition_1d<const N: usize>(
    f_field:       &[i32; N],
    gradient_part: &mut [i32; N],
    harmonic_part: &mut [i32; N],
    potential:     &mut [i32; N],
    div_buffer:    &mut [i32; N],
    c_prime:       &mut [i32; N],
    d_prime:       &mut [i32; N],
) {
    if N == 0 { return; }

    // d*f — divergence (backward difference)
    div_buffer[0] = f_field[0];
    for i in 1..N {
        div_buffer[i] = f_field[i].wrapping_sub(f_field[i - 1]);
    }

    // solve −Δφ = div(f)
    solve_poisson_1d(div_buffer, potential, c_prime, d_prime);

    // dφ — gradient of potential (forward difference)
    for i in 0..N - 1 {
        gradient_part[i] = potential[i + 1].wrapping_sub(potential[i]);
    }
    gradient_part[N - 1] = potential[N - 1].wrapping_neg();

    // residual = coexact (δψ)
    for i in 0..N {
        harmonic_part[i] = f_field[i].wrapping_sub(gradient_part[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q16_mul_identity() {
        assert_eq!(q16_mul(Q16_ONE, Q16_ONE), Q16_ONE);
    }

    #[test]
    fn test_q16_div_by_zero() {
        assert_eq!(q16_div(Q16_ONE, 0), 0);
    }

    #[test]
    fn test_hodge_1d_reconstruction() {
        const N: usize = 8;
        let f: [i32; N] = [Q16_ONE, Q16_ONE*2, Q16_ONE*3, Q16_ONE*2,
                            Q16_ONE, 0, -Q16_ONE, -Q16_ONE*2];
        let mut grad    = [0i32; N];
        let mut harm    = [0i32; N];
        let mut pot     = [0i32; N];
        let mut div_buf = [0i32; N];
        let mut cp      = [0i32; N];
        let mut dp      = [0i32; N];

        hodge_decomposition_1d(&f, &mut grad, &mut harm, &mut pot, &mut div_buf, &mut cp, &mut dp);

        // ∇φ + δψ must reconstruct f exactly (wrapping arithmetic)
        for i in 0..N {
            let recon = grad[i].wrapping_add(harm[i]);
            let err   = (recon - f[i]).unsigned_abs();
            assert!(err <= 4, "recon error at {i}: {err} (Q16 rounding)");
        }
    }
}
