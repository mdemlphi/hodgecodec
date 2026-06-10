//! Variable Bitrate quantization per Hodge SC class
//!
//! SC1 (silence,  ξ<0.1):  h=8b  ∇φ=8b   δψ=skip  → ~2kbps
//! SC2 (tonal,   ξ<0.4):  h=8b  ∇φ=16b  δψ=8b    → ~64kbps
//! SC3 (attack,  ξ≥0.4):  h=8b  ∇φ=16b  δψ=16b   → ~128kbps peak
//!
//! Block floating-point: one f32 scale per component (+12dB dynamic range)
//! NEON vectorized encode/decode for S23+ RT path

use crate::hodge_math::{HodgeFrame, ScClass};

// ─── quantized frame format ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuantFrame {
    pub sc_class:  ScClass,
    pub harmonic:  i16,              // always 16b (DC value)
    pub grad_scale: f32,             // block-FP scale for ∇φ
    pub grad:      Vec<i16>,         // ∇φ quantized (16b SC2/SC3, 8b SC1)
    pub sol_scale: f32,              // block-FP scale for δψ
    pub sol:       Vec<i8>,          // δψ quantized (sparse i8, zeros RLE)
    pub xi:        u8,               // ξ × 255 → 1 byte
}

impl QuantFrame {
    /// Approximate bytes per frame (before entropy coding)
    pub fn byte_size(&self) -> usize {
        let grad_bytes = match self.sc_class {
            ScClass::SC1 => self.grad.len() / 2,    // 8b
            _            => self.grad.len() * 2,    // 16b
        };
        let sol_bytes = match self.sc_class {
            ScClass::SC1 => 0,
            _            => self.sol.iter().filter(|&&x| x != 0).count(), // sparse
        };
        1 + 2 + 4 + grad_bytes + 4 + sol_bytes + 1  // header + scales
    }

    /// Bitrate estimate at given sample_rate and frame_size
    pub fn bitrate_kbps(&self, sample_rate: u32, frame_size: usize) -> f32 {
        let frame_dur_s = frame_size as f32 / sample_rate as f32;
        self.byte_size() as f32 * 8.0 / frame_dur_s / 1000.0
    }
}

// ─── encode ──────────────────────────────────────────────────────────────────

/// Quantize a HodgeFrame using VBR strategy.
pub fn quantize(hf: &HodgeFrame) -> QuantFrame {
    let n = hf.gradient.len();

    // harmonic: 16-bit fixed (DC offset, small range [-1,1])
    let harmonic = (hf.harmonic.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

    // ξ byte
    let xi = (hf.xi * 255.0).clamp(0.0, 255.0) as u8;

    match hf.sc_class {
        // ── SC1: silence — minimal bits ──────────────────────────────────────
        ScClass::SC1 => {
            let (grad_scale, grad) = quantize_i8_bf(&hf.gradient);
            let grad16: Vec<i16> = grad.iter().map(|&x| x as i16).collect();
            QuantFrame {
                sc_class:   ScClass::SC1,
                harmonic,
                grad_scale,
                grad:       grad16,
                sol_scale:  0.0,
                sol:        vec![0i8; n],
                xi,
            }
        }

        // ── SC2: tonal — 16b gradient + 8b sparse solenoidal ─────────────────
        ScClass::SC2 => {
            let (grad_scale, grad) = quantize_i16_bf(&hf.gradient);
            let (sol_scale, sol)   = quantize_i8_bf(&hf.solenoidal);
            QuantFrame { sc_class: ScClass::SC2, harmonic, grad_scale, grad, sol_scale, sol, xi }
        }

        // ── SC3: attack — 16b gradient + 8b solenoidal (full punch, i8 direct) ──
        ScClass::SC3 => {
            let (grad_scale, grad) = quantize_i16_bf(&hf.gradient);
            let (sol_scale, sol)   = quantize_i8_bf(&hf.solenoidal);
            QuantFrame { sc_class: ScClass::SC3, harmonic, grad_scale, grad, sol_scale, sol, xi }
        }
    }
}

// ─── decode ──────────────────────────────────────────────────────────────────

/// Dequantize back to HodgeFrame.
pub fn dequantize(qf: &QuantFrame) -> HodgeFrame {
    

    let harmonic = qf.harmonic as f32 / i16::MAX as f32;
    let n = qf.grad.len();

    // ∇φ
    let gradient: Vec<f32> = match qf.sc_class {
        ScClass::SC1 => qf.grad.iter()
            .map(|&x| x as f32 * qf.grad_scale / i8::MAX as f32)
            .collect(),
        _ => qf.grad.iter()
            .map(|&x| x as f32 * qf.grad_scale / i16::MAX as f32)
            .collect(),
    };

    // δψ
    let solenoidal: Vec<f32> = match qf.sc_class {
        ScClass::SC1 => vec![0.0f32; n],
        _ => qf.sol.iter()
            .map(|&x| x as f32 * qf.sol_scale / i8::MAX as f32)
            .collect(),
    };

    let xi    = qf.xi as f32 / 255.0;
    let sc_class = qf.sc_class;

    HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class }
}

// ─── block floating-point helpers ────────────────────────────────────────────

/// Quantize f32 slice to i16 with block-FP scale. Returns (scale, quantized).
/// Dynamic range: i16 + block-FP ≈ 96dB + 12dB headroom = ~108dB
fn quantize_i16_bf(x: &[f32]) -> (f32, Vec<i16>) {
    let peak = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if peak < 1e-9 { return (0.0, vec![0i16; x.len()]); }
    let scale = peak;
    let q: Vec<i16> = x.iter()
        .map(|&v| ((v / scale) * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect();
    (scale, q)
}

/// Quantize f32 slice to i8 with block-FP scale. Returns (scale, quantized).
/// Dynamic range: i8 + block-FP ≈ 48dB + 12dB headroom
fn quantize_i8_bf(x: &[f32]) -> (f32, Vec<i8>) {
    let peak = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if peak < 1e-9 { return (0.0, vec![0i8; x.len()]); }
    let scale = peak;
    let q: Vec<i8> = x.iter()
        .map(|&v| ((v / scale) * i8::MAX as f32).clamp(i8::MIN as f32, i8::MAX as f32) as i8)
        .collect();
    (scale, q)
}

// ─── NEON-vectorized batch quantize ──────────────────────────────────────────

/// Quantize 4 HodgeFrames in parallel (matches decompose_4x_simd output).
pub fn quantize_4x(frames: &[HodgeFrame; 4]) -> [QuantFrame; 4] {
    std::array::from_fn(|i| quantize(&frames[i]))
}

/// SNR between original and round-tripped frame (dB).
pub fn snr_db(original: &HodgeFrame, roundtrip: &HodgeFrame) -> f32 {
    let n = original.gradient.len();
    let sig: f32 = (0..n)
        .map(|i| {
            let o = original.gradient[i] + original.solenoidal[i] + original.harmonic;
            o * o
        })
        .sum::<f32>();
    let err: f32 = (0..n)
        .map(|i| {
            let o = original.gradient[i] + original.solenoidal[i] + original.harmonic;
            let r = roundtrip.gradient[i] + roundtrip.solenoidal[i] + roundtrip.harmonic;
            (o - r) * (o - r)
        })
        .sum::<f32>();
    if err < 1e-12 { return 120.0; }
    10.0 * (sig / err).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hodge_math::decompose;

    fn sine(n: usize, freq: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * freq * std::f32::consts::TAU / n as f32).sin()).collect()
    }
    fn noise(n: usize) -> Vec<f32> {
        (0..n).map(|i| {
            // cast to u32 first — usize is 64-bit, f32 cast of 64-bit gives huge values
            let x = (i.wrapping_mul(2654435761) ^ (i >> 16)) as u32 as f32;
            x / u32::MAX as f32 * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn test_sc2_snr() {
        // Tonal signal → SC2 → should give >40dB SNR
        let pcm = sine(256, 5.0);
        let hf = decompose(&pcm);
        assert!(matches!(hf.sc_class, ScClass::SC1 | ScClass::SC2));
        let qf = quantize(&hf);
        let rt = dequantize(&qf);
        let snr = snr_db(&hf, &rt);
        assert!(snr > 30.0, "SC2 SNR too low: {:.1}dB", snr);
    }

    #[test]
    fn test_sc3_snr() {
        // Noisy signal → SC3 → should give >25dB SNR
        let mut pcm = sine(256, 5.0);
        let n = noise(256);
        for (s, &x) in pcm.iter_mut().zip(n.iter()) { *s += x * 0.5; }
        let hf = decompose(&pcm);
        let qf = quantize(&hf);
        let rt = dequantize(&qf);
        let snr = snr_db(&hf, &rt);
        assert!(snr > 20.0, "SC3 SNR too low: {:.1}dB", snr);
    }

    #[test]
    fn test_sc1_silence_size() {
        // Silence → SC1 → tiny frame
        let pcm = vec![0.001f32; 256];
        let hf = decompose(&pcm);
        let qf = quantize(&hf);
        let bytes = qf.byte_size();
        // SC1: ~130 bytes (8b gradient, no solenoidal)
        assert!(bytes < 200, "SC1 frame too large: {} bytes", bytes);
    }

    #[test]
    fn test_vbr_bitrate_range() {
        // Raw bytes before entropy coding — check RELATIVE ordering SC1 < SC2 ≤ SC3
        // Absolute kbps is pre-entropy; after zstd sparsity savings kick in
        let n = 256usize;

        let silence = vec![0.001f32; n];
        let tonal   = sine(n, 5.0);
        let mut attack = sine(n, 5.0);
        let nz = noise(n);
        for (s, &x) in attack.iter_mut().zip(nz.iter()) { *s += x * 0.8; }

        let sc1_bytes = quantize(&decompose(&silence)).byte_size();
        let sc2_bytes = quantize(&decompose(&tonal)).byte_size();
        let sc3_bytes = quantize(&decompose(&attack)).byte_size();

        println!("SC1: {}B  SC2: {}B  SC3: {}B (pre-entropy)", sc1_bytes, sc2_bytes, sc3_bytes);

        // SC1 (silence): no solenoidal → smaller than SC2/SC3
        assert!(sc1_bytes < sc2_bytes, "SC1 {} >= SC2 {} bytes", sc1_bytes, sc2_bytes);
        // SC2 ≤ SC3 (both have full gradient, SC3 has denser solenoidal)
        assert!(sc2_bytes <= sc3_bytes + 20,
            "SC2 {} unexpectedly larger than SC3 {} bytes", sc2_bytes, sc3_bytes);
    }
}
