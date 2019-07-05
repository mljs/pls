import Matrix from 'ml-matrix';

import { PLS } from './pls-nipals.js';
import { scale, norm, pow2array, tss } from './utils.js';

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
export function OPLS(x, y, options = {}) {
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


export const oplsWrapper = (data) => {
  let train; let test;

  if (!data.train && !data.test) {
    train = data;
  } else {
    train = data.train;
    test = data.test;
  }

  // get data
  let matrixYData = train.summary().dataMatrix;
  let dataClass = train.getClass()[0].classMatrix;

  if (test) {
    var testMatrixYData = test.summary().dataMatrix;
    var testDataClass = test.getClass()[0].classMatrix;
  }

  // scaling
  let options = { center: true, scale: true };
  matrixYData = scale(matrixYData, options);
  dataClass = scale(dataClass, options);

  if (test) {
    testMatrixYData = scale(testMatrixYData, options);
    testDataClass = scale(testDataClass, options);
  }

  let variables = train.summary().variables;

  if (test) {
    let testVariables = test.summary().variables;
    let testObservations = test.summary().observations;
  }

  let oplsResultV = [];
  let plsCompV = [];
  let testPlsCompV = [];

  let R2Y = []; let R2X = []; let Q2 = [];
  let xResidual, testxResidual, scores, testScores, E_h, testE_h, Yhat, testYhat, oplsResult, plsComp, testPlsComp, msg, testOpls;

  let scoresExport = [];
  let testScoresExport = [];

  for (let i = 1; i < 5; i++) {
    if (i == 1) {
      oplsResult = OPLS(matrixYData.clone(), dataClass.clone());
      (test) ? testOpls = OPLS(testMatrixYData.clone(), testDataClass.clone()) : [];
    } else {
      oplsResult = OPLS(xResidual.clone(), dataClass.clone());
      (test) ? testOpls = OPLS(testxResidual.clone(), testDataClass.clone()) : [];
    }

    oplsResultV.push(oplsResult);
    console.log(oplsResult);
    xResidual = oplsResult.err;
    (test) ? testxResidual = testOpls.err : [];

    plsComp = PLS(xResidual.clone(), dataClass.clone());
    plsCompV.push(plsComp);
    console.log(plsComp);

    if (test) {
      testPlsComp = PLS(testxResidual.clone(), testDataClass.clone());
      testPlsCompV.push(testPlsComp);
    }


    scores = xResidual.mmul(plsComp.weights.transpose()); // ok
    E_h = xResidual.clone().sub(scores.clone().mmul(plsComp.loadings)); // ok
    Yhat = scores.clone().mul(plsComp.betas); // ok

    if (test) {
      testScores = testxResidual.mmul(testPlsComp.weights.transpose()); // ok
      testE_h = testxResidual.clone().sub(testScores.clone().mmul(testPlsComp.loadings)); // ok
      testYhat = testScores.clone().mul(testPlsComp.betas); // ok
    }

    if (test) {
      let testTssy = tss(testDataClass.clone());
      console.log(`tssy:${testTssy}`);
      let testRss = testDataClass.clone().sub(testYhat);

      testRss = testRss.clone().mul(testRss).sum();

      let Q2y = 1 - (testRss / testTssy);
      console.log(`Q2y:${Q2y}`);
      Q2.push(Q2y);
      msg = `Q2y: ${Q2y.toFixed(4).toString()}`;
    }

    let tssy = tss(dataClass.clone());
    let rss = dataClass.clone().sub(Yhat);
    rss = rss.clone().mul(rss).sum();
    let R2y = 1 - (rss / tssy);
    R2Y.push(R2y);
    msg = `R2y: ${R2y.toFixed(4).toString()}`;

    let tssx = tss(matrixYData.clone());
    let X_ex = plsComp.scores.clone().mmul(plsComp.loadings);
    let rssx = tss(X_ex.clone());
    let R2x = rssx / tssx;
    R2X.push(R2x);
    msg = `${msg}\nR2x: ${R2x.toFixed(4).toString()}`;

    // store scores for export
    scoresExport.push({
      scoresX: scores.to1DArray(),
      scoresY: oplsResult.tOrtho.to1DArray()
    });

    if (test) {
      testScoresExport.push({
        scoresX: testScores.map((x) => x[0]),
        scoresY: testOpls.tOrtho.map((x) => x[0])
      });
    }

    console.log(`OPLS iteration over #ofComponents: ${i}`);
  } // end of loop

  let eigenVectorMatrix = new Matrix(plsCompV.length, variables.length);
  plsCompV.forEach((e, i) => eigenVectorMatrix.setRow(i, e.loadings));

  let eigenVectors = new Array(eigenVectorMatrix.rows);

  let xAxis;
  if (typeof (variables[0]) === 'number') {
    xAxis = variables;
  } else {
    xAxis = variables.map((x, i) => i + 1);
  }

  for (let i = 0, l = eigenVectors.length; i < l; i++) {
    eigenVectors[i] = { id: [i + 1],
      eigenVal: null,
      data: { x: xAxis, y: eigenVectorMatrix[i] } };
  }
  /*  API.createData('loadings', eigenVectors);


  let R2YPlot = hlvh.createChart([], R2Y, 'bar');
  API.createData('R2Y', R2YPlot);

  let R2XPlot = hlvh.createChart([], R2X, 'bar');
  API.createData('R2X', R2XPlot);

  if (test) {
    let Q2Plot = hlvh.createChart([], Q2, 'bar');
    API.createData('Q2', Q2Plot);
  }

 */
  return ({
    R2X,
    R2Y,
    scoresExport,
    eigenVectors

    /* getScores(markers, radius) {
      let scoresPlot = [];
      scoresExport.forEach((e, i) => {
        let splot = plot(e.scoresX, e.scoresY, observations, train.getClass()[0], markers, radius);
        scoresPlot.push({ index: i + 1, chart: splot });
      });

      // API.createData('pcaResult', scoresPlot[0].chart.chart);
      // API.createData('pcaResult' + 'ellipse', scoresPlot[0].chart.ellipse);
      // API.createData('scoresPlot', scoresPlot);

      let testScoresPlot = [];
      if (test) {
        testScoresExport.forEach((e, i) => {
          let splot = plot(e.scoresX, e.scoresY, observations, test.getClass()[0], markers, radius * 2);
          testScoresPlot.push({ index: i + 1, chart: splot });
        });

        // API.createData('testPcaResult', testScoresPlot[0].chart.chart);
        // API.createData('testScoresPlot', testScoresPlot);
      }
      if (test) {
        return { scoresPlot, testScoresPlot };
        // return {scoresExport, testScoresExport};
      } else {
        return { scoresPlot };
        // return {scoresExport};
      }

      //   console.log('scores');
      //   let scorePlot = plot(scoresExport[nc - 1].scoresX,
      //     scoresExport[nc - 1].scoresY,
      //     observations,
      //     train.getClass()[0]
      //   );

      //   API.createData('pcaResult', scorePlot.chart);
      //   API.createData('pcaResult' + 'ellipse', scorePlot.ellipse);

      //   return scoresExport;
    } */
  });
};

// ============================================
// Dataset section
// ============================================

export const iris = () => {
  const iris = [
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [4.9, 3, 1.4, 0.2, 'setosa'],
    [4.7, 3.2, 1.3, 0.2, 'setosa'],
    [4.6, 3.1, 1.5, 0.2, 'setosa'],
    [5, 3.6, 1.4, 0.2, 'setosa'],
    [5.4, 3.9, 1.7, 0.4, 'setosa'],
    [4.6, 3.4, 1.4, 0.3, 'setosa'],
    [5, 3.4, 1.5, 0.2, 'setosa'],
    [4.4, 2.9, 1.4, 0.2, 'setosa'],
    [4.9, 3.1, 1.5, 0.1, 'setosa'],
    [5.4, 3.7, 1.5, 0.2, 'setosa'],
    [4.8, 3.4, 1.6, 0.2, 'setosa'],
    [4.8, 3, 1.4, 0.1, 'setosa'],
    [4.3, 3, 1.1, 0.1, 'setosa'],
    [5.8, 4, 1.2, 0.2, 'setosa'],
    [5.7, 4.4, 1.5, 0.4, 'setosa'],
    [5.4, 3.9, 1.3, 0.4, 'setosa'],
    [5.1, 3.5, 1.4, 0.3, 'setosa'],
    [5.7, 3.8, 1.7, 0.3, 'setosa'],
    [5.1, 3.8, 1.5, 0.3, 'setosa'],
    [5.4, 3.4, 1.7, 0.2, 'setosa'],
    [5.1, 3.7, 1.5, 0.4, 'setosa'],
    [4.6, 3.6, 1, 0.2, 'setosa'],
    [5.1, 3.3, 1.7, 0.5, 'setosa'],
    [4.8, 3.4, 1.9, 0.2, 'setosa'],
    [5, 3, 1.6, 0.2, 'setosa'],
    [5, 3.4, 1.6, 0.4, 'setosa'],
    [5.2, 3.5, 1.5, 0.2, 'setosa'],
    [5.2, 3.4, 1.4, 0.2, 'setosa'],
    [4.7, 3.2, 1.6, 0.2, 'setosa'],
    [4.8, 3.1, 1.6, 0.2, 'setosa'],
    [5.4, 3.4, 1.5, 0.4, 'setosa'],
    [5.2, 4.1, 1.5, 0.1, 'setosa'],
    [5.5, 4.2, 1.4, 0.2, 'setosa'],
    [4.9, 3.1, 1.5, 0.2, 'setosa'],
    [5, 3.2, 1.2, 0.2, 'setosa'],
    [5.5, 3.5, 1.3, 0.2, 'setosa'],
    [4.9, 3.6, 1.4, 0.1, 'setosa'],
    [4.4, 3, 1.3, 0.2, 'setosa'],
    [5.1, 3.4, 1.5, 0.2, 'setosa'],
    [5, 3.5, 1.3, 0.3, 'setosa'],
    [4.5, 2.3, 1.3, 0.3, 'setosa'],
    [4.4, 3.2, 1.3, 0.2, 'setosa'],
    [5, 3.5, 1.6, 0.6, 'setosa'],
    [5.1, 3.8, 1.9, 0.4, 'setosa'],
    [4.8, 3, 1.4, 0.3, 'setosa'],
    [5.1, 3.8, 1.6, 0.2, 'setosa'],
    [4.6, 3.2, 1.4, 0.2, 'setosa'],
    [5.3, 3.7, 1.5, 0.2, 'setosa'],
    [5, 3.3, 1.4, 0.2, 'setosa'],
    [7, 3.2, 4.7, 1.4, 'versicolor'],
    [6.4, 3.2, 4.5, 1.5, 'versicolor'],
    [6.9, 3.1, 4.9, 1.5, 'versicolor'],
    [5.5, 2.3, 4, 1.3, 'versicolor'],
    [6.5, 2.8, 4.6, 1.5, 'versicolor'],
    [5.7, 2.8, 4.5, 1.3, 'versicolor'],
    [6.3, 3.3, 4.7, 1.6, 'versicolor'],
    [4.9, 2.4, 3.3, 1, 'versicolor'],
    [6.6, 2.9, 4.6, 1.3, 'versicolor'],
    [5.2, 2.7, 3.9, 1.4, 'versicolor'],
    [5, 2, 3.5, 1, 'versicolor'],
    [5.9, 3, 4.2, 1.5, 'versicolor'],
    [6, 2.2, 4, 1, 'versicolor'],
    [6.1, 2.9, 4.7, 1.4, 'versicolor'],
    [5.6, 2.9, 3.6, 1.3, 'versicolor'],
    [6.7, 3.1, 4.4, 1.4, 'versicolor'],
    [5.6, 3, 4.5, 1.5, 'versicolor'],
    [5.8, 2.7, 4.1, 1, 'versicolor'],
    [6.2, 2.2, 4.5, 1.5, 'versicolor'],
    [5.6, 2.5, 3.9, 1.1, 'versicolor'],
    [5.9, 3.2, 4.8, 1.8, 'versicolor'],
    [6.1, 2.8, 4, 1.3, 'versicolor'],
    [6.3, 2.5, 4.9, 1.5, 'versicolor'],
    [6.1, 2.8, 4.7, 1.2, 'versicolor'],
    [6.4, 2.9, 4.3, 1.3, 'versicolor'],
    [6.6, 3, 4.4, 1.4, 'versicolor'],
    [6.8, 2.8, 4.8, 1.4, 'versicolor'],
    [6.7, 3, 5, 1.7, 'versicolor'],
    [6, 2.9, 4.5, 1.5, 'versicolor'],
    [5.7, 2.6, 3.5, 1, 'versicolor'],
    [5.5, 2.4, 3.8, 1.1, 'versicolor'],
    [5.5, 2.4, 3.7, 1, 'versicolor'],
    [5.8, 2.7, 3.9, 1.2, 'versicolor'],
    [6, 2.7, 5.1, 1.6, 'versicolor'],
    [5.4, 3, 4.5, 1.5, 'versicolor'],
    [6, 3.4, 4.5, 1.6, 'versicolor'],
    [6.7, 3.1, 4.7, 1.5, 'versicolor'],
    [6.3, 2.3, 4.4, 1.3, 'versicolor'],
    [5.6, 3, 4.1, 1.3, 'versicolor'],
    [5.5, 2.5, 4, 1.3, 'versicolor'],
    [5.5, 2.6, 4.4, 1.2, 'versicolor'],
    [6.1, 3, 4.6, 1.4, 'versicolor'],
    [5.8, 2.6, 4, 1.2, 'versicolor'],
    [5, 2.3, 3.3, 1, 'versicolor'],
    [5.6, 2.7, 4.2, 1.3, 'versicolor'],
    [5.7, 3, 4.2, 1.2, 'versicolor'],
    [5.7, 2.9, 4.2, 1.3, 'versicolor'],
    [6.2, 2.9, 4.3, 1.3, 'versicolor'],
    [5.1, 2.5, 3, 1.1, 'versicolor'],
    [5.7, 2.8, 4.1, 1.3, 'versicolor'],
    [6.3, 3.3, 6, 2.5, 'virginica'],
    [5.8, 2.7, 5.1, 1.9, 'virginica'],
    [7.1, 3, 5.9, 2.1, 'virginica'],
    [6.3, 2.9, 5.6, 1.8, 'virginica'],
    [6.5, 3, 5.8, 2.2, 'virginica'],
    [7.6, 3, 6.6, 2.1, 'virginica'],
    [4.9, 2.5, 4.5, 1.7, 'virginica'],
    [7.3, 2.9, 6.3, 1.8, 'virginica'],
    [6.7, 2.5, 5.8, 1.8, 'virginica'],
    [7.2, 3.6, 6.1, 2.5, 'virginica'],
    [6.5, 3.2, 5.1, 2, 'virginica'],
    [6.4, 2.7, 5.3, 1.9, 'virginica'],
    [6.8, 3, 5.5, 2.1, 'virginica'],
    [5.7, 2.5, 5, 2, 'virginica'],
    [5.8, 2.8, 5.1, 2.4, 'virginica'],
    [6.4, 3.2, 5.3, 2.3, 'virginica'],
    [6.5, 3, 5.5, 1.8, 'virginica'],
    [7.7, 3.8, 6.7, 2.2, 'virginica'],
    [7.7, 2.6, 6.9, 2.3, 'virginica'],
    [6, 2.2, 5, 1.5, 'virginica'],
    [6.9, 3.2, 5.7, 2.3, 'virginica'],
    [5.6, 2.8, 4.9, 2, 'virginica'],
    [7.7, 2.8, 6.7, 2, 'virginica'],
    [6.3, 2.7, 4.9, 1.8, 'virginica'],
    [6.7, 3.3, 5.7, 2.1, 'virginica'],
    [7.2, 3.2, 6, 1.8, 'virginica'],
    [6.2, 2.8, 4.8, 1.8, 'virginica'],
    [6.1, 3, 4.9, 1.8, 'virginica'],
    [6.4, 2.8, 5.6, 2.1, 'virginica'],
    [7.2, 3, 5.8, 1.6, 'virginica'],
    [7.4, 2.8, 6.1, 1.9, 'virginica'],
    [7.9, 3.8, 6.4, 2, 'virginica'],
    [6.4, 2.8, 5.6, 2.2, 'virginica'],
    [6.3, 2.8, 5.1, 1.5, 'virginica'],
    [6.1, 2.6, 5.6, 1.4, 'virginica'],
    [7.7, 3, 6.1, 2.3, 'virginica'],
    [6.3, 3.4, 5.6, 2.4, 'virginica'],
    [6.4, 3.1, 5.5, 1.8, 'virginica'],
    [6, 3, 4.8, 1.8, 'virginica'],
    [6.9, 3.1, 5.4, 2.1, 'virginica'],
    [6.7, 3.1, 5.6, 2.4, 'virginica'],
    [6.9, 3.1, 5.1, 2.3, 'virginica'],
    [5.8, 2.7, 5.1, 1.9, 'virginica'],
    [6.8, 3.2, 5.9, 2.3, 'virginica'],
    [6.7, 3.3, 5.7, 2.5, 'virginica'],
    [6.7, 3, 5.2, 2.3, 'virginica'],
    [6.3, 2.5, 5, 1.9, 'virginica'],
    [6.5, 3, 5.2, 2, 'virginica'],
    [6.2, 3.4, 5.4, 2.3, 'virginica'],
    [5.9, 3, 5.1, 1.8, 'virginica']
  ];
  return iris;
};
