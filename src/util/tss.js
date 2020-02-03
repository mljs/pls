/**
 * Get total sum of square
 * @param {Array} x an array
 * @return {Number} - the sum of the squares
 */
export function tss(x) {
  return x
    .clone()
    .mul(x.clone())
    .sum();
}
