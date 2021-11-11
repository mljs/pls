import { Matrix, NIPALS } from 'ml-matrix';

/**
 * OPLS loop
 * @param {Array|Matrix} data matrix with features
 * @param {Array|Matrix} labels an array of labels (dependent variable)
 * @param {Object} [options={}] an object with options
 * @return {Object} an object with model (filteredX: err,
    loadingsXOrtho: pOrtho,
    scoresXOrtho: tOrtho,
    weightsXOrtho: wOrtho,
    weightsPred: w,
    loadingsXpred: p,
    scoresXpred: t,
    loadingsY:)
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
    w = u.transpose().mmul(data).div(u.transpose().mmul(u).get(0, 0));
    w = w.transpose().div(w.norm());
    t = data.mmul(w).div(w.transpose().mmul(w).get(0, 0)); // t_h paso 3

    // calc loading
    c = t.transpose().mmul(labels).div(t.transpose().mmul(t).get(0, 0));

    // calc new u and compare with one in previus iteration (stop criterion)
    uNew = labels.mmul(c.transpose());
    uNew = uNew.div(c.norm() ** 2);
    if (i > 0) {
      diff = uNew.clone().sub(u).pow(2).sum() / uNew.clone().pow(2).sum();
    }
    u = uNew.clone();
  }
  // calc loadings
  let wOrtho;
  let p = t.transpose().mmul(data).div(t.transpose().mmul(t).get(0, 0));
  if (labels.columns > 1) {
    for (let i = 0; i < tW.length - 1; i++) {
      let tw = tW[i].transpose();
      p = p.sub(tw.mmul(p.transpose()).div(tw.mmul(tw.transpose())).mmul(tw));
    }
    wOrtho = p.clone();
  } else {
    wOrtho = p
      .clone()
      .sub(
        w
          .transpose()
          .mmul(p.transpose())
          .div(w.transpose().mmul(w).get(0, 0))
          .mmul(w.transpose()),
      );
  }
  wOrtho.div(wOrtho.norm());
  let tOrtho = data
    .mmul(wOrtho.transpose())
    .div(wOrtho.mmul(wOrtho.transpose()).get(0, 0));

  // orthogonal loadings
  let pOrtho = tOrtho
    .transpose()
    .mmul(data)
    .div(tOrtho.transpose().mmul(tOrtho).get(0, 0));

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

function getWh(xValue, yValue) {
  let result = new Matrix(xValue.columns, yValue.columns);
  for (let i = 0; i < yValue.columns; i++) {
    let yN = yValue.getColumnVector(i).transpose();
    let component = yN.mmul(xValue).div(yN.mmul(yN.transpose()).get(0, 0));
    result.setColumn(i, component.getRow(0));
  }
  return result;
}
