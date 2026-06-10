// hodgecodec — first Hodge decomposition audio codec
// frame = ∇φ (tonal) + δψ (attack) + h (soul/DC)
// © 2026 Maksym Dyachenko <hi@hodgecodec.com> — MIT License

pub mod hodge_math;
pub mod format;
pub mod encoder;
pub mod decoder;
pub mod fingerprint;
pub mod mixer;

/// Q16.16 fixed-point Hodge core — no_std compatible, eBPF-safe.
/// `hodge_fixed::hodge_decomposition_1d` = true Poisson solver (no heap).
pub mod hodge_fixed;
pub mod m202;
pub mod simd;
pub mod windowing;
pub mod quant;

pub use simd::{decompose_4x_simd, decompose_rt, RT_FRAME_SIZE, SIMD_LANES};
pub use windowing::{HannWindow, OlaEncoder, OlaDecoder, encode_ola_simd, decode_ola, HOP};
pub use quant::{quantize, dequantize, quantize_4x, snr_db, QuantFrame};

pub use hodge_math::{decompose, reconstruct, HodgeFrame, ScClass, FRAME_SIZE};
pub use encoder::{encode_wav, encode_wav_compressed, encode_wav_zstd, EncodeStats};
pub use decoder::{decode_hodge, decode_hodge_to_pcm, decode_hodge_with_stats,
                   info_hodge, DecodeParams, DecodeStats};
pub use fingerprint::{fingerprint_hodge, fingerprint_pcm, fingerprint_wav, HodgeFingerprint};
pub use mixer::{mix_hodge, MixParams, MixStats};
pub use m202::{compute_m202, compute_m202_stream, M202Score};

pub mod shazam_baseline;
pub mod watermark;
pub use watermark::{watermark_encode, watermark_decode};
