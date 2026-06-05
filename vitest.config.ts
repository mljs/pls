import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

// ml-matrix ships separate CJS (matrix.js) and ESM (matrix.mjs) builds, and
// some test fixtures (ml-dataset-metadata) bundle an older copy of it. To keep
// a single Matrix class across all packages — so cross-package `instanceof`
// and the entries()-based copy constructor work — alias every `ml-matrix`
// import to the single ESM build, load ml-dataset-metadata from its
// (unbundled) source so it shares that instance, and inline the ml-* deps.
const mlMatrix = fileURLToPath(
  new URL('./node_modules/ml-matrix/matrix.mjs', import.meta.url),
);
const mlDatasetMetadata = fileURLToPath(
  new URL('./node_modules/ml-dataset-metadata/src/index.js', import.meta.url),
);

export default defineConfig({
  resolve: {
    alias: {
      'ml-matrix': mlMatrix,
      'ml-dataset-metadata': mlDatasetMetadata,
    },
  },
  test: {
    globals: true,
    // KOPLS trains on real datasets (in tests and beforeAll hooks); allow ample
    // time under parallel CI load.
    testTimeout: 60_000,
    hookTimeout: 60_000,
    server: {
      deps: {
        inline: ['ml-dataset-metadata'],
      },
    },
    coverage: {
      include: ['src/**/*.js'],
      // istanbul instruments only `include` (src), so the heavy ml-matrix /
      // ml-kernel loops in KOPLS run uninstrumented. With v8's precise coverage
      // they are profiled too, which makes the suite ~8x slower (~99s vs ~13s).
      provider: 'istanbul',
    },
    snapshotFormat: {
      maxOutputLength: Number.MAX_SAFE_INTEGER,
    },
  },
});
