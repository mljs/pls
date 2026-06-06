import { Matrix, NIPALS } from 'ml-matrix';

/**
 * Single OPLS (orthogonal projections to latent structures) NIPALS iteration.
 * Computes the predictive and Y-orthogonal components and returns the data with
 * the orthogonal variation filtered out.
 * @param {Array|Matrix} data - matrix with features (X).
 * @param {Array|Matrix} labels - an array of labels (dependent variable Y).
 * @param {object} [options={}] - an object with options.
 * @param {number} [options.numberOSC=1000] - maximum number of NIPALS iterations.
 * @param {number} [options.limit=1e-10] - convergence threshold used to stop the iteration.
 * @returns {object} the computed model with the following properties:
 * - `filteredX`: X with the orthogonal component removed.
 * - `weightsXOrtho`: Y-orthogonal weights of X.
 * - `loadingsXOrtho`: Y-orthogonal loadings of X.
 * - `scoresXOrtho`: Y-orthogonal scores of X.
 * - `weightsXPred`: predictive weights of X.
 * - `loadingsXpred`: predictive loadings of X.
 * - `scoresXpred`: predictive scores of X.
 * - `loadingsY`: loadings of Y.
 */
export function oplsNipals(data, labels, options = {}) {
  const { numberOSC = 1000, limit = 1e-10 } = options;
  data = Matrix.checkMatrix(data);
  labels = Matrix.checkMatrix(labels);
  let tW = [];
  if (labels.columns > 1) {
    const wh = getWh(data, labels);
    const ssWh = wh.norm() ** 2;
    let ssT = ssWh;
    let pcaW;
    let count = 0;
    do {
      if (count === 0) {
        pcaW = new NIPALS(wh.clone());
        tW.push(pcaW.t);
      } else {
        const data = pcaW.xResidual;
        pcaW = new NIPALS(data);
        tW.push(pcaW.t);
      }
      ssT = pcaW.t.norm() ** 2;
      count++;
    } while (ssT / ssWh > limit);
  }
  let u = labels.getColumnVector(0);
  let diff = 1;
  let t, c, w, uNew;
  for (let i = 0; i < numberOSC && diff > limit; i++) {
    w = u
      .transpose()
      .mmul(data)
      .div(u.norm() ** 2);
    w = w.transpose().div(w.norm());
    t = data.mmul(w).div(w.norm() ** 2); // t_h paso 3

    // calc loading
    c = t
      .transpose()
      .mmul(labels)
      .div(t.norm() ** 2);

    // calc new u and compare with one in previus iteration (stop criterion)
    uNew = labels.mmul(c.transpose()).div(c.norm() ** 2);
    if (i > 0) {
      diff = uNew.clone().sub(u).pow(2).sum() / uNew.clone().pow(2).sum();
    }
    u = uNew.clone();
  }
  // calc loadings
  let wOrtho;
  let p = t
    .transpose()
    .mmul(data)
    .div(t.norm() ** 2);
  if (labels.columns > 1) {
    for (let i = 0; i < tW.length - 1; i++) {
      let tw = tW[i].transpose();
      p = p.sub(
        tw
          .mmul(p.transpose())
          .div(tw.norm() ** 2)
          .mmul(tw),
      );
    }
    wOrtho = p.clone();
  } else {
    wOrtho = p.clone().sub(
      w
        .transpose()
        .mmul(p.transpose())
        .div(w.norm() ** 2)
        .mmul(w.transpose()),
    );
  }
  wOrtho.div(wOrtho.norm());
  let tOrtho = data.mmul(wOrtho.transpose()).div(wOrtho.norm() ** 2);

  // orthogonal loadings
  let pOrtho = tOrtho
    .transpose()
    .mmul(data)
    .div(tOrtho.norm() ** 2);

  // filtered data
  let err = data.clone().sub(tOrtho.mmul(pOrtho));
  return {
    filteredX: err,
    weightsXOrtho: wOrtho,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY: c,
  };
}

/**
 * Computes the W matrix used to initialize the orthogonal loop for multi-column Y.
 * @private
 * @param {Matrix} xValue - the feature matrix (X).
 * @param {Matrix} yValue - the label matrix (Y).
 * @returns {Matrix} the W matrix (one column per Y variable).
 */
function getWh(xValue, yValue) {
  let result = new Matrix(xValue.columns, yValue.columns);
  for (let i = 0; i < yValue.columns; i++) {
    let yN = yValue.getColumnVector(i).transpose();
    let component = yN.mmul(xValue).div(yN.norm() ** 2);
    result.setColumn(i, component.getRow(0));
  }
  return result;
}
