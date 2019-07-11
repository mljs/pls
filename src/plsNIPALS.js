import Matrix from 'ml-matrix';

import { nipals } from './nipals.js';

/**
 * PLS nipals
 * @param {Dataset} dataset a dataset object
 * @param {Matrix} predictions an matrix with predictions
 * @param {Object} options an object with options
 * @return {Object} Object with model
 */

export function plsNIPALS(features, labels,) {
  var X = Matrix.checkMatrix(features.clone());
  var Y = Matrix.checkMatrix(labels.clone());

  var u = Y.getColumnVector(0);
  let ls = nipals(X, Y, u);

  // calculate the X loadings and rescale scores and weights accordingly
  let xP = X.transpose().mmul(ls.t).div(ls.t.transpose().mmul(ls.t).get(0, 0));
  xP = xP.div(xP.norm());

  // calculate beta (regression coefficient) via inverse insted of subtracting q_h directly
  let residual = u.transpose().mmul(ls.t).div(ls.t.transpose().mmul(ls.t).get(0, 0));

  // calc residual matrice X and Y
  let xRes = X.sub(ls.t.clone().mmul(xP.transpose()));
  let yRes = Y.sub(ls.t.clone().mulS(residual.get(0, 0)).mmul(ls.q.transpose()));
  return { xRes, yRes, scores: ls.t, loadings: xP.transpose(), weights: ls.w.transpose(), betas: residual.get(0, 0), qPC: ls.q };
}
