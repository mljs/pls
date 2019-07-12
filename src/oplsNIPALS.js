import Matrix from 'ml-matrix';

import { norm, pow2array } from './utils.js';

let Utils = {};
Utils.norm = function norm(X) {
  return Math.sqrt(X.clone().apply(pow2array).sum());
};

/**
 * OPLS loop
 * @param {Array} x a matrix with features
 * @param {Array} y an array of labels (dependent variable)
 * @param {Object} options an object with options
 * @return {Object} an object with model (filteredX: err,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXOrtho: wOrtho,
    weightsPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY:)
 */
export function oplsNIPALS(x, y, options = {}) {
  const {
    numberOSC = 100,
  } = options;

  let X = Matrix.checkMatrix(x.clone());
  let Y = Matrix.checkMatrix(y.clone());

  let u = Y.getColumnVector(0);

  let diff = 1;
  let t, c, w, uNew;
  for (let i = 0; i < numberOSC && diff > 1e-10; i++) {
    w = u.transpose().mmul(X).div(u.transpose().mmul(u).get(0, 0));
    w = w.transpose().div(norm(w));

    t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));// t_h paso 3

    // calc loading
    c = t.transpose().mmul(Y).div(t.transpose().mmul(t).get(0, 0));

    // calc new u and compare with one in previus iteration (stop criterion)
    uNew = Y.mmul(c.transpose());
    uNew = uNew.div(c.transpose().mmul(c).get(0, 0));

    if (i > 0) {
      diff = uNew.clone().sub(u).pow(2).sum() / uNew.clone().pow(2).sum();
    }

    u = uNew.clone();
  }

  // calc loadings
  let p = t.transpose().mmul(X).div(t.transpose().mmul(t).get(0, 0));

  let wOrtho = p.clone().sub(w.transpose().mmul(p.transpose()).div(w.transpose().mmul(w).get(0, 0)).mmul(w.transpose()));
  wOrtho.div(Utils.norm(wOrtho));

  // orthogonal scores
  let tOrtho = X.mmul(wOrtho.transpose()).div(wOrtho.mmul(wOrtho.transpose()).get(0, 0));

  // orthogonal loadings
  let pOrtho = tOrtho.transpose().mmul(X).div(tOrtho.transpose().mmul(tOrtho).get(0, 0));

  // filtered data
  let err = X.sub(tOrtho.mmul(pOrtho));
  return { filteredX: err,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXOrtho: wOrtho,
    weightsPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY: c };
}

