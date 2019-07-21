
/**
 * PCA NIPALS
 * @param {Matrix} X - a matrix to be factorized
 * @return {Object} - an model (t, w, residual)
 */
export function pcaNIPALS(X) {
  let diff = 1;
  let t, w, tOld;
  t = X.subMatrixColumn([1]);
  let counter = 0;
  for (var i = 0; diff > 1e-10; i++) {
    w = X.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
    w = w.div(w.norm());

    t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));

    if (i > 0) {
      diff = t .clone().sub(tOld).pow(2).sum();
    }

    tOld = t.clone();

    counter++;
    if (counter > 1000) break;
  }
  return { scores: t,
    loadings: w,
    residual: X.clone().sub(t.clone().mmul(w.transpose())) };
}
