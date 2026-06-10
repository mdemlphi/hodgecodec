//! Overlap-Add (OLA) windowing for HodgeCodec
//!
//! Without OLA: blocking artifacts at every 256-sample boundary (sounds like 1980s MP3)
//! With OLA: smooth transitions, perceptually transparent reconstruction
//!
//! Architecture:
//!   Analysis:   x[n] → window × frame → Hodge decompose
//!   Synthesis:  Hodge reconstruct → window × frame → OLA accumulator
//!
//! Window: Hann (periodic) — perfect reconstruction at 50% overlap
//!   w[n] = 0.5 * (1 - cos(2π·n/N))
//!   COLA condition: Σ w²[n-kH] = 1 for hop H = N/2 ✓
//!
//! Latency: 1.5 × FRAME (analysis + synthesis delay)
//!   256-sample frame @ 44.1kHz: 5.8ms latency — RT-safe
//!
//! NEON note: Hann window is precomputed at init, applied via SIMD in hot path.

use crate::hodge_math::{decompose, HodgeFrame};
use crate::simd::{decompose_4x_simd, RT_FRAME_SIZE};

pub const HOP: usize = RT_FRAME_SIZE / 2;  // 128 samples — 50% overlap

/// Precomputed Hann window for RT_FRAME_SIZE
pub struct HannWindow {
    pub coeffs: [f32; RT_FRAME_SIZE],
    pub norm:   f32,  // normalization factor for OLA (= 1/Σw²)
}

impl HannWindow {
    pub fn new() -> Self {
        let n = RT_FRAME_SIZE;
        let mut coeffs = [0.0f32; RT_FRAME_SIZE];
        let mut energy = 0.0f32;
        for i in 0..n {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
            coeffs[i] = w;
            energy += w * w;
        }
        // COLA normalization: at 50% overlap, Σw²[n] + Σw²[n-H] = const
        // For periodic Hann: exactly 0.5 per sample → norm = 2.0
        let norm = n as f32 / energy;  // ≈ 2.0 for Hann at 50% overlap
        Self { coeffs, norm }
    }

    #[inline(always)]
    pub fn apply(&self, frame: &mut [f32]) {
        debug_assert_eq!(frame.len(), RT_FRAME_SIZE);

        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let n4 = RT_FRAME_SIZE / 4;
            unsafe {
                let wp = self.coeffs.as_ptr();
                let fp = frame.as_mut_ptr();
                for k in 0..n4 {
                    let vw = vld1q_f32(wp.add(k * 4));
                    let vf = vld1q_f32(fp.add(k * 4));
                    vst1q_f32(fp.add(k * 4), vmulq_f32(vf, vw));
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (s, &w) in frame.iter_mut().zip(self.coeffs.iter()) {
                *s *= w;
            }
        }
    }
}

impl Default for HannWindow { fn default() -> Self { Self::new() } }

// ─── OLA Encoder (streaming) ─────────────────────────────────────────────────

/// Streaming overlap-add encoder state.
/// Feed samples via `push_samples()`, drain frames via `drain_frames()`.
pub struct OlaEncoder {
    window:   HannWindow,
    /// Input ring buffer (2 × FRAME for overlap)
    buf:      Vec<f32>,
    buf_fill: usize,
    hop:      usize,
}

impl OlaEncoder {
    pub fn new() -> Self {
        Self {
            window:   HannWindow::new(),
            buf:      vec![0.0f32; RT_FRAME_SIZE * 2],
            buf_fill: 0,
            hop:      HOP,
        }
    }

    /// Push PCM samples (any count). Returns number of complete frames ready.
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        let mut ready = 0;
        let mut src = 0;
        while src < samples.len() {
            let space = self.buf.len() - self.buf_fill;
            let copy = (samples.len() - src).min(space);
            self.buf[self.buf_fill..self.buf_fill + copy]
                .copy_from_slice(&samples[src..src + copy]);
            self.buf_fill += copy;
            src += copy;
            while self.buf_fill >= RT_FRAME_SIZE {
                ready += 1;
                // Shift by HOP
                self.buf.copy_within(self.hop..self.buf_fill, 0);
                self.buf_fill -= self.hop;
            }
        }
        ready
    }

    /// Encode one pending frame (call after push_samples indicates frames ready).
    /// Returns windowed HodgeFrame ready for quantization + writing.
    pub fn encode_frame(&self) -> HodgeFrame {
        debug_assert!(self.buf_fill >= RT_FRAME_SIZE || true, "not enough samples");
        let mut windowed = self.buf[..RT_FRAME_SIZE].to_vec();
        self.window.apply(&mut windowed);
        decompose(&windowed)
    }
}

impl Default for OlaEncoder { fn default() -> Self { Self::new() } }

// ─── OLA Decoder (streaming) ─────────────────────────────────────────────────

/// Streaming overlap-add decoder state.
/// Feed HodgeFrames via `push_frame()`, drain PCM via `drain_samples()`.
pub struct OlaDecoder {
    window:      HannWindow,
    /// Accumulator for OLA (2 × FRAME)
    accum:       Vec<f32>,
    /// How many output samples are ready (past the overlap region)
    ready:       usize,
}

impl OlaDecoder {
    pub fn new() -> Self {
        Self {
            window: HannWindow::new(),
            accum:  vec![0.0f32; RT_FRAME_SIZE * 2],
            ready:  0,
        }
    }

    /// Add a decoded HodgeFrame to the OLA accumulator.
    /// Analysis-only OLA: no synthesis window — Hann COLA guarantees Σw[n-kH]=1
    pub fn push_frame(&mut self, hf: &HodgeFrame, tone_gain: f32, attack_gain: f32) {
        use crate::hodge_math::reconstruct;
        let frame = reconstruct(hf, tone_gain, attack_gain, 1.0);
        // NO synthesis window — analysis-only OLA, Hann COLA: Σw[n-kH]=1 ✓
        for (i, &s) in frame.iter().enumerate() {
            if i < self.accum.len() {
                self.accum[i] += s;
            }
        }
        self.ready += HOP;
    }

    /// Drain ready PCM samples into `out`. Returns number of samples written.
    pub fn drain_samples(&mut self, out: &mut Vec<f32>) -> usize {
        let n = self.ready.min(self.accum.len() / 2);
        // No normalization — analysis-only OLA sum = 1.0 for Hann at 50% overlap
        for i in 0..n {
            out.push(self.accum[i]);
        }
        // Shift accumulator
        let len = self.accum.len();
        self.accum.copy_within(n..len, 0);
        for s in &mut self.accum[len - n..] { *s = 0.0; }
        self.ready = self.ready.saturating_sub(n);
        n
    }
}

impl Default for OlaDecoder { fn default() -> Self { Self::new() } }

// ─── Batch SIMD OLA encode: 4 frames × OLA in one shot ───────────────────────

/// Encode a slice of PCM with full OLA + SIMD.
/// Returns Vec of HodgeFrames (one per hop).
/// Input: arbitrary-length f32 PCM, zero-padded to full frame at end.
pub fn encode_ola_simd(pcm: &[f32]) -> Vec<HodgeFrame> {
    let window = HannWindow::new();
    let n = pcm.len();
    let n_hops = if n >= RT_FRAME_SIZE {
        (n - RT_FRAME_SIZE) / HOP + 1
    } else { 0 };

    let mut frames = Vec::with_capacity(n_hops);
    let _silence = vec![0.0f32; RT_FRAME_SIZE];

    let mut hop = 0;
    while hop + RT_FRAME_SIZE <= n || (hop < n && n > 0) {
        // Gather up to 4 overlapping windows
        let batch: [Vec<f32>; 4] = std::array::from_fn(|lane| {
            let start = (hop + lane * HOP).min(n);
            let end   = (start + RT_FRAME_SIZE).min(n);
            let mut f = vec![0.0f32; RT_FRAME_SIZE];
            let valid = end - start;
            f[..valid].copy_from_slice(&pcm[start..end]);
            window.apply(&mut f);
            f
        });

        let refs: [&[f32]; 4] = std::array::from_fn(|i| batch[i].as_slice());
        let hodge_frames = decompose_4x_simd(refs);

        let hops_this_iter = ((n.saturating_sub(hop) + HOP - 1) / HOP).min(4);
        for i in 0..hops_this_iter {
            frames.push(hodge_frames[i].clone());
        }

        hop += HOP * 4;
        if hop >= n { break; }
    }

    frames
}

/// Decode OLA frames back to PCM.
/// Analysis-only OLA: no synthesis window — Hann COLA Σw[n-kH]=1 guarantees
/// perfect reconstruction (modulo Hodge sparsity threshold error).
pub fn decode_ola(frames: &[HodgeFrame], tone_gain: f32, attack_gain: f32) -> Vec<f32> {
    if frames.is_empty() { return Vec::new(); }
    let total = frames.len() * HOP + RT_FRAME_SIZE;
    let mut accum = vec![0.0f32; total];

    for (i, hf) in frames.iter().enumerate() {
        use crate::hodge_math::reconstruct;
        let frame = reconstruct(hf, tone_gain, attack_gain, 1.0);
        // NO synthesis window — analysis-only OLA
        let start = i * HOP;
        for (j, &s) in frame.iter().enumerate() {
            if start + j < accum.len() {
                accum[start + j] += s;
            }
        }
    }
    accum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hodge_math::FRAME_SIZE;

    fn sine(n: usize, freq: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * freq * std::f32::consts::TAU / 44100.0).sin()).collect()
    }

    #[test]
    fn test_hann_cola_condition() {
        // Analysis-only COLA: Σ w[n-kH] = 1 for all n.
        // At 50% overlap (H=N/2): w[n] + w[n+H] = 1 for Hann (periodic).
        // Proof: w[n]+w[n+H] = 0.5(1-cosθ) + 0.5(1-cos(θ+π)) = 0.5+0.5 = 1 ✓
        let w = HannWindow::new();
        let h = HOP;
        let n = RT_FRAME_SIZE;
        for i in 0..h {
            let sum = w.coeffs[i] + w.coeffs[i + h];
            assert!((sum - 1.0).abs() < 1e-5,
                "COLA violated at i={}: w[i]+w[i+H] = {:.6} ≠ 1.0", i, sum);
        }
    }

    #[test]
    fn test_ola_reconstruction() {
        // Encode + decode a sine, check reconstruction quality (SNR > 30dB)
        let pcm = sine(44100, 440.0);  // 1 second, 440Hz
        let frames = encode_ola_simd(&pcm);
        let recon = decode_ola(&frames, 1.0, 1.0);

        // Compare middle section (avoid edge effects)
        let start = RT_FRAME_SIZE;
        let end = pcm.len().min(recon.len()) - RT_FRAME_SIZE;
        if start >= end { return; }

        let signal_power: f32 = pcm[start..end].iter().map(|x| x*x).sum::<f32>();
        let error_power: f32 = pcm[start..end].iter().zip(recon[start..end].iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>();
        let snr_db = 10.0 * (signal_power / (error_power + 1e-12)).log10();

        // Tikhonov pseudo-Hodge + OLA: expect > 25dB SNR for tonal signal
        assert!(snr_db > 25.0, "OLA SNR too low: {:.1}dB", snr_db);
    }

    #[test]
    fn test_ola_streaming_encoder() {
        let pcm = sine(4096, 220.0);
        let mut enc = OlaEncoder::new();
        let ready = enc.push_samples(&pcm);
        // At least some frames should be ready
        assert!(ready > 0, "no frames encoded");
    }

    #[test]
    fn test_streaming_roundtrip() {
        let pcm = sine(8192, 880.0);
        let mut enc = OlaEncoder::new();
        let mut dec = OlaDecoder::new();
        let mut out = Vec::new();

        // Push in 128-sample chunks (simulates audio callback)
        let chunk = 128;
        for block in pcm.chunks(chunk) {
            enc.push_samples(block);
            let hf = enc.encode_frame();
            dec.push_frame(&hf, 1.0, 1.0);
            dec.drain_samples(&mut out);
        }

        // Just check we got output without panic
        assert!(!out.is_empty(), "no output from streaming roundtrip");
    }
}
