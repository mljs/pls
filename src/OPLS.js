import { Matrix, NIPALS } from 'ml-matrix';

import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, summaryMetadata, getFolds } from './utils.js';

export class OPLS {
  constructor(features, labels, options = {}) {
    const {
      nComp = 3,
      center = true,
      scale = true,
      cvFolds = [],
    } = options;


    const dataClass = summaryMetadata(labels).classMatrix;
    const dataLabels = labels;

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    this.Q2 = [];
    this.model = [];
    this.predictedScores = [];
    this.orthScores = [];
    this.predictions = [];

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
    let xres;

    let ncCounter = 0;
    for (let i = 1; i < nComp + 1; i++) {
      let totalPred = new Matrix(dataClass.rows, 1);
      let scorePred = new Matrix(dataClass.rows, 1);
      let scoreOrtho = new Matrix(dataClass.rows, 1);

      let counter = 0;
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

        if (i === 1) {
          oplsResult = oplsNIPALS(Xk, Yk);
          xresCv.push(oplsResult.filteredX);
        } else {
          oplsResult = oplsNIPALS(xresCv[counter], Yk);
        }
        xresCv[counter] = oplsResult.filteredX;
        counter++;

        let plsComp = new NIPALS(oplsResult.filteredX, { Y: Yk });

        let Eh = testXk;
        // removing the orthogonal components from PLS
        let scores;
        for (let idx = 0; idx < i; idx++) {
          // console.log('idx', idx);
          scores = Eh.clone().mmul(oplsResult.weightsXOrtho.transpose()); // ok
          Eh.sub(scores.clone().mmul(oplsResult.loadingsXOrtho));
        }
        // prediction
        let tPred = Eh.clone().mmul(plsComp.w.transpose());
        // this should be summed over ncomp (pls_prediction.R line 23)
        let Yhat = tPred.clone().mmul(plsComp.betas);

        // adding all prediction from all folds
        for (let i = 0; i < k.testIndex.length; i++) {
          totalPred.setRow(k.testIndex[i], [Yhat.get(i, 0)]);
          scorePred.setRow(k.testIndex[i], [tPred.get(i, 0)]);
          scoreOrtho.setRow(k.testIndex[i], [scores.get(i, 0)]);
        }
      } // end of loop over folds

      this.predictedScores.push(scorePred.to1DArray()); // could be done as a matrix
      this.orthScores.push(scoreOrtho.to1DArray());
      this.predictions.push(totalPred.to1DArray());
      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implmented (check opls.R line 183) TODO
      let tssy = tss(dataClass.center('column').scale('column'));
      let press = tss(dataClass.clone().sub(totalPred));
      let Q2y = 1 - (press / tssy);
      this.Q2.push(Q2y);

      // calculate the R2y for the complete data
      if (i === 1) {
        modelNcomp = this.predictAll(matrixYData, dataClass);
        // oplsC = oplsNIPALS(features, labels);
      } else {
        modelNcomp = this.predictAll(xres, dataClass);
        // oplsC = oplsNIPALS(xres, labels);
      }
      // Deflated matrix for next compoment
      xres = modelNcomp.xRes;
      // Let last pls model for output
      plsC = modelNcomp.plsC;
      modelNcomp.Q2y = this.Q2;
      this.model.push(modelNcomp);
      // console.log('model', this.model);
      console.log(`OPLS iteration over # of Components: ${i}`);
      ncCounter++;
    } // end of loop

    let tCV = this.predictedScores; // could be done as a matrix
    let tOrthCV = this.orthScores;
    let tOrth = this.model[ncCounter - 1].tOrth;
    let pOrth = this.model[ncCounter - 1].pOrth;
    let wOrth = this.model[ncCounter - 1].wOrth;
    let XOrth = this.model[ncCounter - 1].XOrth;
    let FeaturesCS = matrixYData.center('column').scale('column');
    let labelsCS = dataClass.center('column').scale('column');
    let Xres = FeaturesCS.clone().sub(XOrth);
    let plsCall = new NIPALS(Xres, { Y: labelsCS });
    let Q2y = this.Q2;
    let R2x = this.model.map((x) => x.R2x);
    let R2y = this.model.map((x) => x.R2y);

    let E = Xres.clone().sub(plsCall.t.clone().mmul(plsCall.p));
    this.output = { Q2y,
      R2x,
      R2y,
      tPred: plsC.t.to1DArray(), // ok
      pPred: plsC.p.to1DArray(), // ok
      wPred: plsC.w.to1DArray(), // ok
      betasPred: plsC.betas, // ok
      Qpc: plsC.q, // ok
      tCV,
      tOrthCV,
      tOrth, // ok
      pOrth, // ok
      wOrth, // ok
      XOrth, // ok
      Yres: plsC.yResidual.to1DArray(),
      E }; // ok
  }
  predictAll(features, labels) {
    features.center('column').scale('column');
    labels.center('column').scale('column');
    // console.log('features', features);
    let oplsC = oplsNIPALS(features, labels);


    let plsC = new NIPALS(oplsC.filteredX, { Y: labels });

    let tPred = oplsC.filteredX.clone().mmul(plsC.w.transpose());
    let Yhat = tPred.clone().mmul(plsC.betas);

    let tssy = tss(labels);
    // console.log('tssy', tssy);
    let rss = tss(labels.clone().sub(Yhat));

    let R2y = 1 - (rss / tssy);

    let xEx = plsC.t.clone().mmul(plsC.p.clone());
    let rssx = tss(xEx);
    let tssx = tss(features.clone());
    // console.log('tssx', tssx);
    let R2x = (rssx / tssx);


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
    let scoresX = this.predictedScores.map((x) => x.to1DArray());
    let scoresY = this.orthScores.map((x) => x.to1DArray());
    return { scoresX, scoresY };
  }
  getPredictions() {
    let predictions = this.predictions.map((x) => x.to1DArray());
    return predictions;
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
