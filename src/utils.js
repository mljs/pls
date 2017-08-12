import Matrix from 'ml-matrix';
import Stat from 'ml-stat/matrix';

/**
 * Function that given vector, returns his norm
 * @param {Vector} X
 * @returns {number} Norm of the vector
 */
export function norm(X) {
    return Math.sqrt(X.clone().apply(pow2array).sum());
}

/**
 * Function that pow 2 each element of a Matrix or a Vector,
 * used in the apply method of the Matrix object
 * @param i - index i.
 * @param j - index j.
 * @return The Matrix object modified at the index i, j.
 * */
export function pow2array(i, j) {
    this[i][j] = this[i][j] * this[i][j];
    return this;
}

/**
 * Function that normalize the dataset and return the means and
 * standard deviation of each feature.
 * @param dataset
 * @returns {{result: Matrix, means: (*|number), std: Matrix}} dataset normalized, means
 *                                                             and standard deviations
 */
export function featureNormalize(dataset) {
    var means = Stat.mean(dataset);
    var std = Stat.standardDeviation(dataset, means, true);
    var result = Matrix.checkMatrix(dataset).subRowVector(means);
    return {result: result.divRowVector(std), means: means, std: std};
}
