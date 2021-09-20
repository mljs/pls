import ConfusionMatrix from 'ml-confusion-matrix';
import { getFolds } from 'ml-cross-validation';
import { Matrix, NIPALS } from 'ml-matrix';
import { getRocCurve, getAuc } from 'ml-roc-multiclass';

import { oplsNipals } from './oplsNipals.js';
import { tss } from './util/tss.js';

/**
 * Creates new OPLS (orthogonal partial latent structures) from features and labels.
 * @param {Array} data - matrix containing data (X).
 * @param {Array} labels - 1D Array containing metadata (Y).
 * @param {Object} [options={}]
 * @param {boolean} [options.center = true] - should the data be centered (subtract the mean).
 * @param {boolean} [options.scale = true] - should the data be scaled (divide by the standard deviation).
 * @param {Array} [options.cvFolds = []] - Allows to provide folds as array of objects with the arrays trainIndex and testIndex as properties.
 * @param {number} [options.nbFolds = 7] - Allows to generate the defined number of folds with the training and test set choosen randomly from the data set.
 * */

export class OPLS {
  constructor(data, labels, options = {}) {
    if (data === true) {
      const opls = options;
      this.center = opls.center;
      this.scale = opls.scale;
      this.means = opls.means;
      this.meansY = opls.meansY;
      this.stdevs = opls.stdevs;
      this.stdevs = opls.stdevsY;
      this.model = opls.model;
      this.tCV = opls.tCV;
      this.tOrthCV = opls.tOrthCV;
      this.yHatCV = opls.yHatCV;
      this.mode = opls.mode;
      return;
    }

    const features = new Matrix(data);
    // set default values
    // cvFolds allows to define folds for testing purpose
    const { center = true, scale = true, cvFolds = [], nbFolds = 7 } = options;

    let group;
    if (typeof labels[0] === 'number') {
      // numeric labels: OPLS regression is used
      this.mode = 'regression';
      group = Matrix.from1DArray(labels.length, 1, labels);
    } else if (typeof labels[0] === 'string') {
      // non-numeric labels: OPLS-DA is used
      this.mode = 'discriminantAnalysis';
      group = Matrix.checkMatrix(createDummyY(labels)).transpose();
    }

    // getting center and scale the features (all)
    this.center = center;
    if (this.center) {
      this.means = features.mean('column');
      this.meansY = group.mean('column');
    } else {
      this.stdevs = null;
    }
    this.scale = scale;
    if (this.scale) {
      this.stdevs = features.standardDeviation('column');
      this.stdevsY = group.standardDeviation('column');
    } else {
      this.means = null;
    }

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    let folds;
    if (cvFolds.length > 0) {
      folds = cvFolds;
    } else {
      folds = getFolds(labels, nbFolds);
    }
    const Q2 = [];
    const aucResult = [];
    this.model = [];

    this.tCV = [];
    this.tOrthCV = [];
    this.yHatCV = [];
    const oplsCV = [];

    let modelNC = [];

    // this code could be made more efficient by reverting the order of the loops
    // this is a legacy loop to be consistent with R code from MetaboMate package
    // this allows for having statistic (R2) from CV to decide wether to continue
    // with more latent structures
    let overfitted = false;
    let nc = 0;
    let value;

    do {
      const yHatk = new Matrix(group.rows, 1);
      const tPredk = new Matrix(group.rows, 1);
      const tOrthk = new Matrix(group.rows, 1);
      oplsCV[nc] = [];
      for (let f = 0; f < folds.length; f++) {
        const trainTest = this._getTrainTest(features, group, folds[f]);
        const testXk = trainTest.testFeatures;
        const Xk = trainTest.trainFeatures;
        const Yk = trainTest.trainLabels;

        // determine center and scale of training set
        const dataCenter = Xk.mean('column');
        const dataSD = Xk.standardDeviation('column');

        // center and scale training set
        if (center) {
          Xk.center('column');
          Yk.center('column');
        }

        if (scale) {
          Xk.scale('column');
          Yk.scale('column');
        }

        // perform opls
        let oplsk;
        if (nc === 0) {
          oplsk = oplsNipals(Xk, Yk);
        } else {
          oplsk = oplsNipals(oplsCV[nc - 1][f].filteredX, Yk);
        }

        // store model for next component
        oplsCV[nc][f] = oplsk;
        const plsCV = new NIPALS(oplsk.filteredX, { Y: Yk });

        // scaling the test dataset with respect to the train
        testXk.center('column', { center: dataCenter });
        testXk.scale('column', { scale: dataSD });

        const Eh = testXk;
        // removing the orthogonal components from PLS
        let scores;
        for (let idx = 0; idx < nc + 1; idx++) {
          scores = Eh.mmul(oplsCV[idx][f].weightsXOrtho.transpose()); // ok
          Eh.sub(scores.mmul(oplsCV[idx][f].loadingsXOrtho));
        }

        // prediction
        const tPred = Eh.mmul(plsCV.w.transpose());
        const yHatComponents = tPred
          .mmul(plsCV.betas)
          .mmul(plsCV.q.transpose()); // ok
        const yHat = new Matrix(yHatComponents.rows, 1);
        for (let i = 0; i < yHatComponents.rows; i++) {
          yHat.setRow(i, [yHatComponents.getRowVector(i).sum()]);
        }
        // adding all prediction from all folds
        for (let i = 0; i < folds[f].testIndex.length; i++) {
          yHatk.setRow(folds[f].testIndex[i], [yHat.get(i, 0)]);
          tPredk.setRow(folds[f].testIndex[i], [tPred.get(i, 0)]);
          tOrthk.setRow(folds[f].testIndex[i], [scores.get(i, 0)]);
        }
      } // end of loop over folds

      this.tCV.push(tPredk);
      this.tOrthCV.push(tOrthk);
      this.yHatCV.push(yHatk);

      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implemented (check opls.R line 183) TODO
      const tssy = tss(group.center('column').scale('column'));
      let press = 0;
      for (let i = 0; i < group.columns; i++) {
        press += tss(group.getColumnVector(i).sub(yHatk));
      }
      const Q2y = 1 - press / group.columns / tssy;
      Q2.push(Q2y);
      if (this.mode === 'regression') {
        value = Q2y;
      } else if (this.mode === 'discriminantAnalysis') {
        const rocCurve = getRocCurve(labels, yHatk.to1DArray());
        const areaUnderCurve = getAuc(rocCurve);
        aucResult.push(areaUnderCurve);
        value = areaUnderCurve;
      }

      // calculate the R2y for the complete data
      if (nc === 0) {
        modelNC = this._predictAll(features, group);
      } else {
        modelNC = this._predictAll(modelNC.xRes, group, {
          scale: false,
          center: false,
        });
      }

      // adding the predictive statistics from CV
      let listOfValues;
      modelNC.Q2y = Q2;
      if (this.mode === 'regression') {
        listOfValues = Q2;
      } else {
        listOfValues = aucResult;
        modelNC.auc = aucResult;
      }
      modelNC.value = value;

      if (nc > 0) {
        overfitted = listOfValues[nc - 1] >= value ? true : false;
      }
      this.model.push(modelNC);
      // store the model for each component
      nc++;
      // console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
    } while (!overfitted); // end of loop over nc
    // store scores from CV
    const tCV = this.tCV;
    const tOrthCV = this.tOrthCV;
    const yHatCV = this.yHatCV;
    const m = this.model[nc - 1];
    const XOrth = m.XOrth;
    const FeaturesCS = features.center('column').scale('column');
    let labelsCS;
    if (this.mode === 'regression') {
      labelsCS = group.center('column').scale('column');
    } else {
      labelsCS = group;
    }
    const Xres = FeaturesCS.clone().sub(XOrth);
    const plsCall = new NIPALS(Xres, { Y: labelsCS });
    const E = Xres.clone().sub(plsCall.t.mmul(plsCall.p));

    const R2x = this.model.map((x) => x.R2x);
    const R2y = this.model.map((x) => x.R2y);

    this.output = {
      Q2y: Q2,
      auc: aucResult,
      R2x,
      R2y,
      tPred: m.plsC.t,
      pPred: m.plsC.p,
      wPred: m.plsC.w,
      betasPred: m.plsC.betas,
      Qpc: m.plsC.q,
      tCV,
      tOrthCV,
      yHatCV,
      oplsCV,
      tOrth: m.tOrth,
      pOrth: m.pOrth,
      wOrth: m.wOrth,
      XOrth,
      yHat: m.totalPred,
      Yres: m.plsC.yResidual,
      E,
    };
  }

  /**
   * get access to all the computed elements
   * Mainly for debug and testing
   * @return {Object} output object
   */
  getLogs() {
    return this.output;
  }

  getScores() {
    const scoresX = this.tCV.map((x) => x.to1DArray());
    const scoresY = this.tOrthCV.map((x) => x.to1DArray());
    return { scoresX, scoresY };
  }

  /**
   * Load an OPLS model from JSON
   * @param {Object} model
   * @return {OPLS}
   */
  static load(model) {
    if (typeof model.name !== 'string') {
      throw new TypeError('model must have a name property');
    }
    if (model.name !== 'OPLS') {
      throw new RangeError(`invalid model: ${model.name}`);
    }
    return new OPLS(true, [], model);
  }

  /**
   * Export the current model to a JSON object
   * @return {Object} model
   */
  toJSON() {
    return {
      name: 'OPLS',
      center: this.center,
      scale: this.scale,
      means: this.means,
      stdevs: this.stdevs,
      model: this.model,
      tCV: this.tCV,
      tOrthCV: this.tOrthCV,
      yHatCV: this.yHatCV,
    };
  }

  /**
   * Predict scores for new data
   * @param {Matrix} features - a matrix containing new data
   * @param {Object} [options]
   * @param {Array} [options.trueLabel] - an array with true values to compute confusion matrix
   * @param {Number} [options.nc] - the number of components to be used
   * @return {Object} - predictions
   */
  predict(newData, options = {}) {
    const { trueLabels = [] } = options;
    let labels;
    if (trueLabels.length > 0) {
      labels = Matrix.from1DArray(trueLabels.length, 1, trueLabels);
    }

    const features = new Matrix(newData);

    // scaling the test dataset with respect to the train
    if (this.center) {
      features.center('column', { center: this.means });
      if (labels.rows > 0 && this.mode === 'regression') {
        labels.center('column', { center: this.meansY });
      }
    }
    if (this.scale) {
      features.scale('column', { scale: this.stdevs });
      if (labels.rows > 0 && this.mode === 'regression') {
        labels.scale('column', { scale: this.stdevsY });
      }
    }

    let nc;
    if (this.mode === 'regression') {
      nc = this.model[0].Q2y.length;
    } else {
      nc = this.model[0].auc.length;
    }

    const Eh = features.clone();
    // removing the orthogonal components from PLS
    let tOrth;
    let wOrth;
    let pOrth;
    let yHat;
    let tPred;
    for (let idx = 0; idx < nc; idx++) {
      wOrth = this.model[idx].wOrth.transpose();
      pOrth = this.model[idx].pOrth;
      tOrth = Eh.mmul(wOrth);
      Eh.sub(tOrth.mmul(pOrth));
      // prediction
      tPred = Eh.mmul(this.model[idx].plsC.w.transpose());
      yHat = tPred.mmul(this.model[idx].plsC.betas);
    }

    if (labels.rows > 0) {
      if (this.mode === 'regression') {
        const tssy = tss(labels);
        const press = tss(labels.clone().sub(yHat));
        const Q2y = 1 - press / tssy;

        return { tPred, tOrth, yHat, Q2y };
      } else if (this.mode === 'discriminantAnalysis') {
        const confusionMatrix = ConfusionMatrix.fromLabels(
          trueLabels,
          yHat.to1DArray(),
        );

        const rocCurve = getRocCurve(trueLabels, yHat.to1DArray());
        const auc = getAuc(rocCurve);
        return { tPred, tOrth, yHat, confusionMatrix, auc };
      }
    } else {
      return { tPred, tOrth, yHat };
    }
  }

  _predictAll(features, labels, options = {}) {
    // cannot use the global this.center here
    // since it is used in the NC loop and
    // centering and scaling should only be
    // performed once
    const { center = true, scale = true } = options;

    if (center) {
      features.center('column');
      labels.center('column');
    }

    if (scale) {
      features.scale('column');
      labels.scale('column');
      // reevaluate tssy and tssx after scaling
      // must be global because re-used for next nc iteration
      // tssx is only evaluate the first time
      this.tssy = tss(labels);
      this.tssx = tss(features);
    }

    const oplsC = oplsNipals(features, labels);
    const plsC = new NIPALS(oplsC.filteredX, { Y: labels });

    const tPred = oplsC.filteredX.mmul(plsC.w.transpose());
    const yHatComponents = tPred.mmul(plsC.betas).mmul(plsC.q.transpose()); // ok
    const yHat = new Matrix(yHatComponents.rows, 1);
    for (let i = 0; i < yHatComponents.rows; i++) {
      yHat.setRow(i, [yHatComponents.getRowVector(i).sum()]);
    }
    let rss = 0;
    for (let i = 0; i < labels.columns; i++) {
      rss += tss(labels.getColumnVector(i).sub(yHat));
    }
    const R2y = 1 - rss / labels.columns / this.tssy;
    const xEx = plsC.t.mmul(plsC.p);
    const rssx = tss(xEx);
    const R2x = rssx / this.tssx;

    return {
      R2y,
      R2x,
      xRes: oplsC.filteredX,
      tOrth: oplsC.scoresXOrtho,
      pOrth: oplsC.loadingsXOrtho,
      wOrth: oplsC.weightsXOrtho,
      tPred: tPred,
      totalPred: yHat,
      XOrth: oplsC.scoresXOrtho.mmul(oplsC.loadingsXOrtho),
      oplsC,
      plsC,
    };
  }
  /**
   *
   * @param {*} X - dataset matrix object
   * @param {*} group - labels matrix object
   * @param {*} index - train and test index (output from getFold())
   */
  _getTrainTest(X, group, index) {
    const testFeatures = new Matrix(index.testIndex.length, X.columns);
    const testLabels = new Matrix(index.testIndex.length, group.columns);
    index.testIndex.forEach((el, idx) => {
      testFeatures.setRow(idx, X.getRow(el));
      testLabels.setRow(idx, group.getRow(el));
    });

    const trainFeatures = new Matrix(index.trainIndex.length, X.columns);
    const trainLabels = new Matrix(index.trainIndex.length, group.columns);
    index.trainIndex.forEach((el, idx) => {
      trainFeatures.setRow(idx, X.getRow(el));
      trainLabels.setRow(idx, group.getRow(el));
    });

    return {
      trainFeatures,
      testFeatures,
      trainLabels,
      testLabels,
    };
  }
}

function createDummyY(array) {
  const features = [...new Set(array)];
  const result = [];
  if (features.length > 2) {
    for (let i = 0; i < features.length; i++) {
      const feature = [];
      for (let j = 0; j < array.length; j++) {
        const point = features[i] === array[j] ? 1 : -1;
        feature.push(point);
      }
      result.push(feature);
    }
    return result;
  } else {
    const result = [];
    for (let j = 0; j < array.length; j++) {
      const point = features[0] === array[j] ? 2 : 1;
      result.push(point);
    }
    return [result];
  }
}
