import Matrix from 'ml-matrix';

import { plsNIPALS } from './plsNIPALS.js';
import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, summaryMetadata, getFolds } from './utils.js';

export class OPLS {
  constructor(features, labels, options = {}) {
    if (features === true) {
      this.model = options.model;
      return;
    }
    const {
      nComp = 3,
      center = true,
      scale = true,
      cvFolds = [],
      obsLabels = Array(features.rows).fill(null).map((x, i) => `OBS${i + 1}`),
      varLabels = Array(features.columns).fill(null).map((x, i) => `VAR${i + 1}`),
    } = options;

    const matrixYData = features;
    const dataClass = summaryMetadata(labels).classMatrix;
    const dataLabels = labels;

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    this.Q2 = [];
    this.model = [];

    let folds;
    if (cvFolds.length > 0) {
      folds = cvFolds;
    } else {
      folds = getFolds(dataLabels, 5);
    }

    let modelNcomp = [];

    for (let i = 1; i < nComp; i++) {
      console.log('ITERATION ncomp', i);

      let totalPred = new Matrix(dataClass.rows, 1);
      let scorePred = new Matrix(dataClass.rows, 1);

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

        let oplsResult;
        // start opls
        if (i === 1) {
          oplsResult = oplsNIPALS(Xk, Yk);
        } else {
          oplsResult = oplsNIPALS(oplsResult.filteredX, Yk);
        }

        let plsComp = plsNIPALS(oplsResult.filteredX, Yk);

        let Eh = testXk;
        // removing the orthogonal components from PLS
        for (let idx = 0; idx < i; idx++) {
          let scores = Eh.clone().mmul(oplsResult.weightsXOrtho.transpose()); // ok
          Eh.sub(scores.clone().mmul(oplsResult.loadingsXOrtho));
        }
        // prediction
        let tPred = Eh.clone().mmul(plsComp.weights.transpose());
        // this should be summed over ncomp (pls_prediction.R line 23)
        let Yhat = tPred.clone().mul(plsComp.betas);

        // adding all prediction from all folds
        for (let i = 0; i < k.testIndex.length; i++) {
          totalPred.setRow(k.testIndex[i], [Yhat.get(i, 0)]);
          scorePred.setRow(k.testIndex[i], [tPred.get(i, 0)]);
        }
      } // end of loop over folds

      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implmented (check opls.R line 183) TODO
      let tssy = tss(dataClass.center('column').scale('column'));
      let press = tss(dataClass.clone().sub(totalPred));
      let Q2y = 1 - (press / tssy);
      this.Q2.push({ i, Q2y });
      console.log('Q2', this.Q2);

      // calculate the R2y for the complete data
      if (i === 1) {
        modelNcomp = this.predictAll(matrixYData, dataClass);
      } else {
        modelNcomp = this.predictAll(modelNcomp.xRes, dataClass);
      }
      this.model.push(modelNcomp);
      console.log('model', this.model);
      console.log(`OPLS iteration over #ofComponents: ${i}`);
    } // end of loop
  }
  predictAll(features, labels) {
    let oplsC = oplsNIPALS(features, labels);
    let plsC = plsNIPALS(oplsC.filteredX, labels);

    let tPred = oplsC.filteredX.clone().mmul(plsC.weights.transpose());
    let Yhat = tPred.clone().mul(plsC.betas);
    let tssy = tss(labels);
    let rss = tss(labels.clone().sub(Yhat));

    let R2y = 1 - (rss / tssy);

    let xEx = plsC.scores.clone().mmul(plsC.loadings.clone());
    let rssx = tss(xEx);
    let tssx = tss(features);
    let R2x = (rssx / tssx);

    let resX = oplsC.filteredX;

    return { R2y, R2x, resX, scorePred: tPred, totalPred: Yhat, oplsC, plsC };
  }
  static model() {
    return this.model;
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
