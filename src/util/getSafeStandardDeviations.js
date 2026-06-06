/**
 * Computes the per-column standard deviations of a matrix, replacing any zero
 * (a constant column) by `1`.
 *
 * Scaling a column by its standard deviation is a division; for a constant
 * column the standard deviation is `0`, so the division yields `Infinity`/`NaN`
 * which then poisons every downstream computation. Replacing the zero by `1`
 * leaves the already-centered constant column at all-zeros, which is the
 * neutral, information-free contribution expected from such a column.
 * @private
 * @param {import('ml-matrix').Matrix} matrix - the matrix whose columns are scaled.
 * @returns {number[]} the per-column standard deviations, with every zero replaced by `1`.
 */
export function getSafeStandardDeviations(matrix) {
  const standardDeviations = matrix.standardDeviation('column');
  for (let i = 0; i < standardDeviations.length; i++) {
    if (standardDeviations[i] === 0) standardDeviations[i] = 1;
  }
  return standardDeviations;
}
