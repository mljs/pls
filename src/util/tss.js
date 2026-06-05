import { Matrix } from 'ml-matrix';

/**
 * Get total sum of square
 * @param {Array} x - an array
 * @returns {number} - the sum of the squares
 */
export function tss(x) {
  return Matrix.mul(x, x).sum();
}
