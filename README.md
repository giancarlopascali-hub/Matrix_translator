# Matrix PCA Studio

Matrix PCA Studio converts a library of equally sized matrices into normalized PCA coordinates for Bayesian optimization, then reconstructs optimizer candidates with the exact matching PCA basis. Binary matrices are decoded with the strict rule `value > threshold ? 1 : 0`.

## Workflow

1. Load at least two varying matrices in the encoder.
2. Review the measured round-trip accuracy and exact-matrix rate. Increase the requested `K` if the representation is too lossy.
3. Download both the latent CSV and basis JSON. They share a `model_id` and must remain paired.
4. Use only the `z1 ... zK` columns as bounded `[0, 1]` Bayesian-optimizer parameters. Objective columns may be appended without becoming latent coordinates.
5. Upload the optimizer's candidate CSV and the matching basis JSON in the decoder, in either order.

PCA is compact and continuous, but it cannot losslessly encode every possible binary matrix with a small `K`. Exact reconstruction of the fitted library is possible when the retained components span that library's affine variation. The app reports the real result instead of silently adding unsupported dimensions.

Basis files use the versioned `pca-linear-v2` contract. Older basis files are rejected because they used incompatible fallback components and normalization bounds; re-encode the source matrices to create a current CSV/basis pair.

## Development

```bash
npm install
npm run dev
npm run lint
npm run test
npm run build
```

The generated Python/NumPy adapter loads the exported JSON basis directly. It does not refit PCA, so its decoding math stays aligned with the web app.
