import Matrix from 'ml-matrix';

import { norm } from './utils.js';

/**
 * PLS nipals
 * @param {Dataset} dataset a dataset object
 * @param {Matrix} predictions an matrix with predictions
 * @param {Object} options an object with options
 * @return {Object} Object with model
 */
export function PLS(dataset, predictions, options = {}) {
  const {
    numberOSC = 2,
    scale = false
  } = options;

  var X = Matrix.checkMatrix(dataset);
  var Y = Matrix.checkMatrix(predictions);

  if (scale) {
    X = featureNormalize(X).result;
    Y = featureNormalize(y).result;
  }
  // console.log(X, Y);
  var rows = X.rows;
  var columns = X.columns;

  var u = Y.getColumnVector(0);
  let diff = 1;
  let t, q, w, tOld;
  for (var i = 0; i < numberOSC && diff > 1e-10; i++) {
    w = X.transpose().mmul(u).div(u.transpose().mmul(u).get(0, 0));
    // console.log('w without norm', JSON.stringify(w));
    w = w.div(norm(w));
    // console.log('w', JSON.stringify(w));
    // console.log(w.transpose().mmul(w));
    // calc X scores
    t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));// t_h paso 3
    // calc loading
    // console.log('scores', t);
    if (i > 0) {
      diff = t.clone().sub(tOld).pow(2).sum();
      // console.log('diff', diff);
    }
    tOld = t.clone();
    // Y block, calc weights, normalise and calc Y scores
    // steps can be omitted for 2 class Y (simply by setting q_h=1)
    q = Y.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
    q = q.div(norm(q));

    u = Y.mmul(q).div(q.transpose().mmul(q).get(0, 0));
    console.log(`PLS iteration: ${i}`);
  }
  // calculate the X loadings and rescale scores and weights accordingly
  let xP = X.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
  xP = xP.div(norm(xP));
  // calc Y loadings
  let yP = Y.transpose().mmul(u).div(u.transpose().mmul(u).get(0, 0));
  // calc b for residuals Y
  // calculate beta (regression coefficient) via inverse insted of subtracting q_h directly
  let residual = u.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
  // console.log('residuals', residual);
  // calc residual matrice X and Y
  let xRes = X.sub(t.clone().mmul(xP.transpose()));
  let yRes = Y.sub(t.clone().mulS(residual.get(0, 0)).mmul(q.transpose()));
  return { xRes, yRes, scores: t, loadings: xP.transpose(), weights: w.transpose(), betas: residual.get(0, 0), qPC: q };
}
