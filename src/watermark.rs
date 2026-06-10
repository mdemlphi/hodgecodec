// Hodge Watermark — LSB steganography in WAV audio
// Author string embedded into sample LSBs with CRC32 verification
// Inaudible: ±1 LSB change on i16 = 0.003dB RMS change

use std::fs::File;
use std::io::{BufReader, BufWriter};
use hound::{WavReader, WavWriter, WavSpec, SampleFormat};

const MAGIC: &[u8; 8] = b"HODGEWM1";
const MAGIC_BITS: usize = 64;
const LEN_BITS: usize = 16;  // max 65535 char author string
const HEADER_BITS: usize = MAGIC_BITS + LEN_BITS;

// Layout: samples[0..32] = stride (u32, LSB per sample, stride=1)
//         samples[32 + i*stride] = payload bits (MAGIC+LEN+DATA+CRC32)

/// Embed author string into WAV LSBs. Output WAV is bit-identical except LSBs.
pub fn watermark_encode(input: &str, output: &str, author: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(input)?;
    let spec = reader.spec();
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;

    let author_bytes = author.as_bytes();
    let crc = crc32(author_bytes);
    let mut payload_bytes = Vec::with_capacity(8 + 2 + author_bytes.len() + 4);
    payload_bytes.extend_from_slice(MAGIC);
    payload_bytes.push((author_bytes.len() >> 8) as u8);
    payload_bytes.push((author_bytes.len() & 0xFF) as u8);
    payload_bytes.extend_from_slice(author_bytes);
    payload_bytes.extend_from_slice(&crc.to_be_bytes());

    let payload_bits = payload_bytes.len() * 8;
    let available = samples.len().saturating_sub(32);
    if available < payload_bits {
        return Err(format!("WAV too short: need ≥{} samples, have {}",
            32 + payload_bits, samples.len()).into());
    }

    // Stride: spread evenly across available space
    let stride = (available / payload_bits).max(1);

    let mut out = samples.clone();

    // Encode stride in first 32 samples (u32 BE, LSB per sample)
    for i in 0..32 {
        let bit = ((stride as u32) >> (31 - i)) & 1;
        out[i] = (out[i] & !1) | (bit as i16);
    }

    // Encode payload at offset 32 with computed stride
    let mut bit_idx = 0usize;
    for byte in &payload_bytes {
        for b in (0..8).rev() {
            let bit = (byte >> b) & 1;
            let pos = 32 + bit_idx * stride;
            if pos < out.len() {
                out[pos] = (out[pos] & !1) | (bit as i16);
            }
            bit_idx += 1;
        }
    }

    let wspec = WavSpec { channels: spec.channels, sample_rate: spec.sample_rate,
        bits_per_sample: spec.bits_per_sample, sample_format: SampleFormat::Int };
    let mut writer = WavWriter::create(output, wspec)?;
    for s in &out { writer.write_sample(*s)?; }
    writer.finalize()?;

    Ok(author_bytes.len())
}

/// Extract author string from WAV LSBs. Returns (author, verified).
pub fn watermark_decode(input: &str) -> Result<(String, bool), Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(input)?;
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;

    if samples.len() < 32 + MAGIC_BITS { return Err("File too short".into()); }

    // Read stride from first 32 samples
    let mut stride_val: u32 = 0;
    for i in 0..32 {
        stride_val = (stride_val << 1) | ((samples[i] & 1) as u32);
    }
    let stride = stride_val as usize;
    if stride == 0 || stride > samples.len() { return Err("No watermark found".into()); }

    // Read MAGIC
    let magic_read: Vec<u8> = (0..8).map(|bi| read_byte_at(&samples, 32, bi, stride)).collect();
    if &magic_read[..] != MAGIC { return Err("No watermark found".into()); }

    // Read length
    let len_hi = read_byte_at(&samples, 32, 8, stride);
    let len_lo = read_byte_at(&samples, 32, 9, stride);
    let author_len = ((len_hi as usize) << 8) | (len_lo as usize);
    if author_len == 0 || author_len > 4096 { return Err("Invalid watermark length".into()); }

    // Read author + CRC
    let mut author_bytes = Vec::with_capacity(author_len);
    for i in 0..author_len {
        author_bytes.push(read_byte_at(&samples, 32, 10 + i, stride));
    }
    let crc_stored = u32::from_be_bytes([
        read_byte_at(&samples, 32, 10 + author_len,     stride),
        read_byte_at(&samples, 32, 10 + author_len + 1, stride),
        read_byte_at(&samples, 32, 10 + author_len + 2, stride),
        read_byte_at(&samples, 32, 10 + author_len + 3, stride),
    ]);
    let verified = crc_stored == crc32(&author_bytes);
    Ok((String::from_utf8_lossy(&author_bytes).into_owned(), verified))
}

fn read_byte_at(samples: &[i16], offset: usize, byte_idx: usize, stride: usize) -> u8 {
    let mut b = 0u8;
    for bit_i in 0..8 {
        let pos = offset + (byte_idx * 8 + (7 - bit_i)) * stride;
        if pos < samples.len() {
            b |= ((samples[pos] & 1) as u8) << bit_i;
        }
    }
    b
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 { crc = (crc >> 1) ^ 0xEDB88320; }
            else { crc >>= 1; }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_stable() {
        assert_eq!(crc32(b"MDIACH.EML"), crc32(b"MDIACH.EML"));
        assert_ne!(crc32(b"MDIACH.EML"), crc32(b"other"));
    }

    #[test]
    fn test_watermark_roundtrip() {
        // Create minimal WAV in memory via temp files
        let tmp_in  = "/tmp/wm_test_in.wav";
        let tmp_out = "/tmp/wm_test_out.wav";
        let author  = "MDIACH.EML";

        // Write 44100 samples of silence
        let spec = WavSpec { channels: 1, sample_rate: 44100,
            bits_per_sample: 16, sample_format: SampleFormat::Int };
        let mut w = WavWriter::create(tmp_in, spec).unwrap();
        for _ in 0..44100 { w.write_sample(0i16).unwrap(); }
        w.finalize().unwrap();

        watermark_encode(tmp_in, tmp_out, author).unwrap();
        let (decoded, verified) = watermark_decode(tmp_out).unwrap();
        assert_eq!(decoded, author);
        assert!(verified, "CRC32 verification failed");
    }
}
