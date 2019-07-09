import Matrix from 'ml-matrix';

import { norm, pow2array } from './utils.js';

let Utils = {};
Utils.norm = function norm(X) {
  return Math.sqrt(X.clone().apply(pow2array).sum());
};

/**
 * @private
 * normalize dataset
 * @param {Dataset} dataset a dataset object
 * @return {Object} an object with a scaled dataset
 */
function featureNormalize(dataset) {
  let means = dataset.mean('column');
  let std = dataset.standardDeviation('column');
  let result = Matrix.checkMatrix(dataset)
    .subRowVector(means)
    .divRowVector(std);
  return { result: result, means: means, std: std };
}

/**
 * OPLS loop
 * @param {Array} x a dataset object
 * @param {Array} y an array with responses (dependent variable)
 * @param {Object} options an object with options
 */
export function oplsNIPALS(x, y, options = {}) {
  const {
    numberOSC = 100,
    scale = true,
  } = options;

  let X = Matrix.checkMatrix(x);
  let Y = Matrix.checkMatrix(y);

  if (scale) {
    X = featureNormalize(X).result;
    Y = featureNormalize(Y).result;
  }

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
    console.log(`OPLS iteration: ${i}`);
  }

  // calc loadings
  let p = t.transpose().mmul(X).div(t.transpose().mmul(t).get(0, 0));

  let wOrtho = p.clone().sub(w.transpose().mmul(p.transpose()).div(w.transpose().mmul(w).get(0, 0)).mmul(w.transpose()));
  wOrtho.div(Utils.norm(wOrtho));

  // orthogonal scores
  let tOrtho = X.mmul(wOrtho.transpose()).div(wOrtho.transpose().mmul(wOrtho).get(0, 0));

  // orthogonal loadings
  let pOrtho = tOrtho.transpose().mmul(X).div(tOrtho.transpose().mmul(tOrtho).get(0, 0));

  // filtered data
  let err = X.sub(tOrtho.mmul(pOrtho));
  return { err, pOrtho, tOrtho, wOrtho, w, p, t, c };
}

