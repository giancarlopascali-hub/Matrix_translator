# Matrix PCA Studio

Matrix PCA Studio converts a library of equally sized matrices into normalized PCA coordinates for Bayesian optimization, then reconstructs optimizer candidates with the exact matching PCA basis. Binary matrices are decoded with the strict rule `value > threshold ? 1 : 0`.

## Workflow

1. Load one or more matrices in the encoder. Every encoded row contains exactly the requested `K` values (up to the mathematical limit `K <= rows × columns`). With one matrix, the app uses a hidden contrasting anchor to learn one direction, then completes the basis to `K`; the anchor is never exported as a matrix and is discarded automatically when a second real matrix is loaded.
2. Review the measured round-trip accuracy and exact-matrix rate. Increase the requested `K` if the representation is too lossy.
3. Download both the latent CSV and basis JSON. They share a `model_id` and must remain paired.
4. Use only the `z1 ... zK` columns as bounded `[0, 1]` Bayesian-optimizer parameters. Objective columns may be appended without becoming latent coordinates.
5. Upload the optimizer's candidate CSV and the matching basis JSON in the decoder, in either order.
6. For binary output, open Design Validation to require one orthogonally connected region of `1` cells and all-one `2×2` blocks in at least two distinct matrix corners. Export the supplied latent CSV with an added `validation` column containing `0` or `1`.

The latent CSV includes comment metadata that identifies data-informed and completion dimensions, recommends the default optimization set, and records fixed reference values for completion dimensions. Coordinate values are exported as readable decimals rather than floating-point noise such as `4e-28`.

The basis separates data-informed PCA dimensions from deterministic orthogonal completion dimensions. This keeps the CSV and decoder contract fixed at exactly `K` values even when only a few matrices are available, while reporting how many directions were actually learned from the data. The completion dimensions are valid decoder directions that let the optimizer explore outside the fitted library; they do not claim additional explained variance.

PCA is compact and continuous, but it cannot losslessly encode every possible binary matrix with a small `K`. Exact reconstruction of the fitted library is possible when `K` spans that library's affine variation. The encoder reports measured round-trip accuracy and exact-match rate so lossy configurations remain visible.

Basis files use the versioned `pca-linear-v2` contract. Older basis files are rejected because they used incompatible fallback components and normalization bounds; re-encode the source matrices to create a current CSV/basis pair.

## Development

```bash
pnpm install --frozen-lockfile
pnpm run dev
pnpm run lint
pnpm run test
pnpm run build
```

The generated Python/NumPy adapter loads the exported JSON basis directly. It does not refit PCA, so its decoding math stays aligned with the web app.

Uploaded matrices can be removed individually from the encoder list. The basis and latent CSV are refitted immediately using the remaining matrices.
