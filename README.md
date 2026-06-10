# HodgeCodec

**First audio codec based on Hodge decomposition and de Rham cohomology.**

## Theory

Any audio signal W decomposes into three orthogonal components:

```
W = ∇φ + δψ + h
```

- `∇φ` — tonal/harmonic gradient (smooth, sustained tones)
- `δψ` — attack/transient coexact component
- `h ∈ H¹(M,ℝ)` — harmonic invariant (topological fingerprint)

The component `h` is invariant under pitch shifts, time stretching, and MP3 compression.

## Results

- Cosine similarity ≥ 0.96 for matched tracks
- 10/11 wins vs Shazam baseline
- Training-free, O(N) time
- Implemented in open-source Rust

## Preprint

**Topological Audio Fingerprinting via Hodge Decomposition**  
Dyachenko, M. (2026). Zenodo.  
DOI: [10.5281/zenodo.20618759](https://doi.org/10.5281/zenodo.20618759)

## License

MIT © 2026 Maksym Dyachenko
