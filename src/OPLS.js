import { Matrix, NIPALS } from 'ml-matrix';

import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, getFolds } from './utils.js';

export class OPLS {
  constructor(features, labels, options = {}) {
    const {
      nComp = 3,
      center = true,
      scale = true,
      cvFolds = [],
    } = options;

    if (typeof (labels[0]) === 'number') {
      console.warn('numeric labels: OPLS regression is used');
      var dataLabels = labels;
      var dataClass = Matrix
        .from1DArray(labels.length, 1, labels);
    } else if (typeof (labels[0]) === 'string') {
      console.warn('non-numeric labels: OPLS-DA is used');
    }

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    let folds;
    if (cvFolds.length > 0) {
      folds = cvFolds;
    } else {
      folds = getFolds(dataLabels, 5);
    }

    let matrixYData = features;
    let oplsResult;
    let xresCv = [];
    let plsC;
    let modelNcomp = [];

    let Q2 = [];
    this.model = [];
    this.PredictedScoresNC = [];
    this.orthScoresNC = [];
    this.totalPredNC = [];
    let oplsResultNC = [];

    for (var nc = 0; nc < nComp; nc++) {
      let totalPredCV = new Matrix(dataClass.rows, 1);
      let PredictedScoresCV = new Matrix(dataClass.rows, 1);
      let orthoScoresCV = new Matrix(dataClass.rows, 1);
      let oplsResultCV = [];

      let Kcounter = 0;
      for (let k of folds) {
        let testXk = getTrainTest(matrixYData, dataClass, k).testFeatures;

        let Xk = getTrainTest(matrixYData, dataClass, k).trainFeatures;
        let Yk = getTrainTest(matrixYData, dataClass, k).trainLabels;

        // determine center and scale of training set
        let dataCenter = Xk.mean('column');
        let dataSD = Xk.standardDeviation('column');

        // center and scale training set
        if (center) {
          Xk.center('column');
          Yk.center('column');
        }

        if (scale) {
          Xk.scale('column');
          Yk.scale('column');
        }

        // scaling the test dataset with respect to the train
        testXk.center('column', { center: dataCenter });
        testXk.scale('column', { scale: dataSD });

        if (nc === 0) {
          oplsResult = oplsNIPALS(Xk, Yk);
        } else {
          oplsResult = oplsNIPALS(xresCv[Kcounter], Yk);
        }
        xresCv[Kcounter] = oplsResult.filteredX;
        oplsResultCV[Kcounter] = oplsResult;
        oplsResultNC[nc] = oplsResultCV;


        let plsComp = new NIPALS(oplsResult.filteredX, { Y: Yk });

        let Eh = testXk;
        // removing the orthogonal components from PLS
        let scores;

        for (let idx = 0; idx < nc + 1; idx++) {
          // scores = Eh.clone().mmul(oplsResult.weightsXOrtho.transpose()); // ok
          scores = Eh.clone().mmul(oplsResultNC[idx][Kcounter].weightsXOrtho.transpose()); // ok
          Eh.sub(scores.clone().mmul(oplsResultNC[idx][Kcounter].loadingsXOrtho));
        }

        // prediction
        let tPred = Eh.clone().mmul(plsComp.w.transpose());
        // this should be summed over ncomp (pls_prediction.R line 23)
        let Yhat = tPred.clone().mmul(plsComp.betas); // ok

        // adding all prediction from all folds
        for (let i = 0; i < k.testIndex.length; i++) {
          totalPredCV.setRow(k.testIndex[i], [Yhat.get(i, 0)]);
          PredictedScoresCV.setRow(k.testIndex[i], [tPred.get(i, 0)]);
          orthoScoresCV.setRow(k.testIndex[i], [scores.get(i, 0)]);
        }
        Kcounter++;
      } // end of loop over folds

      this.PredictedScoresNC.push(PredictedScoresCV.to1DArray()); // could be done as a matrix
      this.orthScoresNC.push(orthoScoresCV.to1DArray());
      this.totalPredNC.push(totalPredCV.to1DArray());

      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implemented (check opls.R line 183) TODO
      let tssy = tss(dataClass.center('column').scale('column'));
      let press = tss(dataClass.clone().sub(totalPredCV));
      let Q2y = 1 - (press / tssy);
      Q2.push(Q2y); // ok

      // calculate the R2y for the complete data
      if (nc === 0) {
        modelNcomp = this.predictAll(matrixYData, dataClass);
      } else {
        modelNcomp = this.predictAll(modelNcomp.xRes,
          dataClass,
          options = { scale: false, center: false });
      }
      // Deflated matrix for next compoment
      // Let last pls model for output
      plsC = modelNcomp.plsC;
      modelNcomp.Q2y = Q2;
      this.model.push(modelNcomp);

      console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
    } // end of loop over nc

    let tCV = this.PredictedScoresNC;
    let tOrthCV = this.orthScoresNC;
    let tOrth = this.model[nc - 1].tOrth;
    let pOrth = this.model[nc - 1].pOrth;
    let wOrth = this.model[nc - 1].wOrth;
    let XOrth = this.model[nc - 1].XOrth;
    let FeaturesCS = matrixYData.center('column').scale('column');
    let labelsCS = dataClass.center('column').scale('column');
    let Xres = FeaturesCS.clone().sub(XOrth);
    let plsCall = new NIPALS(Xres, { Y: labelsCS });
    let Q2y = Q2;
    let R2x = this.model.map((x) => x.R2x);
    let R2y = this.model.map((x) => x.R2y);

    let E = Xres.clone().sub(plsCall.t.clone().mmul(plsCall.p));

    this.output = { Q2y, // ok
      R2x, // ok
      R2y, // ok
      tPred: plsC.t.to1DArray(),
      pPred: plsC.p.to1DArray(),
      wPred: plsC.w.to1DArray(),
      betasPred: plsC.betas,
      Qpc: plsC.q,
      tCV, // ok
      tOrthCV, // ok
      tOrth,
      pOrth,
      wOrth,
      XOrth,
      Yres: plsC.yResidual.to1DArray(),
      E };
  }

  predictAll(features, labels, options = {}) {
    const { center = true,
      scale = true } = options;

    if (center) {
      features.center('column');
      labels.center('column');
    }

    if (scale) {
      features.scale('column');
      labels.scale('column');
      this.tssy = tss(labels);
      this.tssx = tss(features);
    }

    let oplsC = oplsNIPALS(features, labels);

    let plsC = new NIPALS(oplsC.filteredX, { Y: labels });

    let tPred = oplsC.filteredX.clone().mmul(plsC.w.transpose());
    let Yhat = tPred.clone().mmul(plsC.betas);

    let rss = tss(labels.clone().sub(Yhat));

    let R2y = 1 - (rss / this.tssy);

    let xEx = plsC.t.clone().mmul(plsC.p.clone());
    let rssx = tss(xEx);

    let R2x = (rssx / this.tssx);

    return { R2y,
      R2x,
      xRes: oplsC.filteredX,
      tOrth: oplsC.scoresXOrtho.to1DArray(),
      pOrth: oplsC.loadingsXOrtho.to1DArray(),
      wOrth: oplsC.weightsXOrtho.to1DArray(),
      tPred: tPred.to1DArray(),
      totalPred: Yhat.to1DArray(),
      XOrth: oplsC.scoresXOrtho.clone().mmul(oplsC.loadingsXOrtho),
      oplsC,
      plsC };
  }
  summary(options = {}) {
    const { idx } = options;
    if (!idx) {
      return this.model;
    } else {
      return this.model[idx];
    }
  }
  getScores() {
    let scoresX = this.PredictedScoresNC.map((x) => x.to1DArray());
    let scoresY = this.orthScoresNC.map((x) => x.to1DArray());
    return { scoresX, scoresY };
  }
  gettotalPredNC() {
    let totalPredNC = this.totalPredNC.map((x) => x.to1DArray());
    return totalPredNC;
  }
  getResults() {
    return this.output;
  }
}

function getTrainTest(matrixYData, dataClass, index) {
  let testFeatures = new Matrix(index.testIndex.length, matrixYData.columns);
  let testLabels = new Matrix(index.testIndex.length, 1);
  index.testIndex.forEach((el, idx) => {
    testFeatures.setRow(idx, matrixYData.getRow(el));
    testLabels.setRow(idx, dataClass.getRow(el));
  });

  let trainFeatures = new Matrix(index.trainIndex.length, matrixYData.columns);
  let trainLabels = new Matrix(index.trainIndex.length, 1);
  index.trainIndex.forEach((el, idx) => {
    trainFeatures.setRow(idx, matrixYData.getRow(el));
    trainLabels.setRow(idx, dataClass.getRow(el));
  });

  return ({
    trainFeatures,
    testFeatures,
    trainLabels,
    testLabels
  });
}
