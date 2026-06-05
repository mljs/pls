import { Matrix } from 'ml-matrix';

/**
 * Given a vector, returns its norm.
 * @private
 * @param {Matrix} X - the vector.
 * @returns {number} norm of the vector.
 */
export function norm(X) {
  return Math.sqrt(X.clone().apply(pow2array).sum());
}

/**
 * Powers by 2 each element of a Matrix or a Vector, used in the apply method of
 * the Matrix object.
 * @private
 * @param {number} i - index i.
 * @param {number} j - index j.
 */
export function pow2array(i, j) {
  // eslint-disable-next-line no-invalid-this -- `this` is the Matrix bound by apply()
  this.set(i, j, this.get(i, j) ** 2);
}

/**
 * Normalizes the dataset and returns the means and standard deviation of each
 * feature.
 * @private
 * @param {Matrix} dataset - the dataset to normalize.
 * @returns {object} dataset normalized, means and standard deviations.
 */
export function featureNormalize(dataset) {
  let means = dataset.mean('column');
  let std = dataset.standardDeviation('column', {
    mean: means,
    unbiased: true,
  });
  let result = Matrix.checkMatrix(dataset).subRowVector(means);
  return { result: result.divRowVector(std), means, std };
}

/**
 * Initializes an array of matrices.
 * @private
 * @param {Array} array - the array to initialize.
 * @param {boolean} isMatrix - whether the array holds 2D matrices.
 * @returns {Array} array with the matrices initialized.
 */
export function initializeMatrices(array, isMatrix) {
  if (isMatrix) {
    for (let i = 0; i < array.length; ++i) {
      for (let j = 0; j < array[i].length; ++j) {
        let elem = array[i][j];
        array[i][j] = elem !== null ? new Matrix(array[i][j]) : undefined;
      }
    }
  } else {
    for (let i = 0; i < array.length; ++i) {
      array[i] = new Matrix(array[i]);
    }
  }

  return array;
}
