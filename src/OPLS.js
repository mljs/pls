import { Matrix, NIPALS } from 'ml-matrix';
import ConfusionMatrix from 'ml-confusion-matrix';
// import { getTrainTest } from 'ml-cross-validation';

import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, getFolds } from './utils.js';

export class OPLS {
  constructor(features, labels, options = {}) {
    if (features === true) {
      const opls = options;
      this.center = opls.center;
      this.scale = opls.scale;
      this.means = opls.means;
      this.stdevs = opls.stdevs;
      this.model = opls.model;
      this.tCV = this.tCV;
      this.tOrthCV = this.tOrthCV;
      return;
    }

    console.log(typeof labels);

    const {
      nComp = 3,
      center = true,
      scale = true,
      cvFolds = [],
    } = options;

    this.center = center;
    if (this.center) {
      this.means = features.mean('column');
    } else {
      this.stdevs = null;
    }
    this.scale = scale;
    if (this.scale) {
      this.stdevs = features.standardDeviation('column');
    } else {
      this.means = null;
    }

    if (typeof (labels.get(0, 0)) === 'number') {
      console.warn('numeric labels: OPLS regression is used');
      // var group = Matrix
      //   .from1DArray(labels.length, 1, labels);
      var group = labels;
      console.log(group);
    } else if (typeof (labels[0]) === 'string') {
      console.warn('non-numeric labels: OPLS-DA is used');
    }

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    let folds;
    if (cvFolds.length > 0) {
      folds = cvFolds;
    } else {
      folds = getFolds(labels, 5);
    }

    let filteredXCV = [];
    let modelNC = [];
    let Q2 = [];

    let yHatNC = [];
    let oplsNC = [];

    this.tCV = [];
    this.tOrthCV = [];
    this.model = [];

    for (var nc = 0; nc < nComp; nc++) {
      let yHatCV = new Matrix(group.rows, 1);
      let tPredCV = new Matrix(group.rows, 1);
      let scoresCV = new Matrix(group.rows, 1);
      let oplsCV = [];

      let fold = 0;
      for (let f of folds) {
        let trainTest = this._getTrainTest(features, group, f);
        let testXk = trainTest.testFeatures;
        let Xk = trainTest.trainFeatures;
        let Yk = trainTest.trainLabels;

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

        if (nc === 0) {
          oplsCV[fold] = oplsNIPALS(Xk, Yk);
        } else {
          oplsCV[fold] = oplsNIPALS(filteredXCV[fold], Yk);
        }
        filteredXCV[fold] = oplsCV[fold].filteredX;
        oplsNC[nc] = oplsCV;

        let plsCV = new NIPALS(oplsCV[fold].filteredX, { Y: Yk });

        // scaling the test dataset with respect to the train
        testXk.center('column', { center: dataCenter });
        testXk.scale('column', { scale: dataSD });

        let Eh = testXk;
        // removing the orthogonal components from PLS
        let scores;
        for (let idx = 0; idx < nc + 1; idx++) {
          scores = Eh.clone().mmul(oplsNC[idx][fold].weightsXOrtho.transpose()); // ok
          Eh.sub(scores.clone().mmul(oplsNC[idx][fold].loadingsXOrtho));
        }

        // prediction
        let tPred = Eh.clone().mmul(plsCV.w.transpose());
        // this should be summed over ncomp (pls_prediction.R line 23)
        let yHat = tPred.clone().mmul(plsCV.betas); // ok

        // adding all prediction from all folds
        for (let i = 0; i < f.testIndex.length; i++) {
          yHatCV.setRow(f.testIndex[i], [yHat.get(i, 0)]);
          tPredCV.setRow(f.testIndex[i], [tPred.get(i, 0)]);
          scoresCV.setRow(f.testIndex[i], [scores.get(i, 0)]);
        }
        fold++;
      } // end of loop over folds

      this.tCV.push(tPredCV);
      this.tOrthCV.push(scoresCV);
      yHatNC.push(yHatCV);

      // calculate Q2y for all the prediction (all folds)
      // ROC for DA is not implemented (check opls.R line 183) TODO
      let tssy = tss(group.center('column').scale('column'));
      let press = tss(group.clone().sub(yHatCV));
      let Q2y = 1 - (press / tssy);
      Q2.push(Q2y); // ok

      // calculate the R2y for the complete data
      if (nc === 0) {
        modelNC = this._predictAll(features, group);
      } else {
        modelNC = this._predictAll(modelNC.xRes,
          group,
          options = { scale: false, center: false });
      }
      // Deflated matrix for next compoment
      // Let last pls model for output

      modelNC.Q2y = Q2;
      this.model.push(modelNC);
      console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
    } // end of loop over nc

    // store scores from CV
    let tCV = this.tCV;
    let tOrthCV = this.tOrthCV;

    let m = this.model[nc - 1];
    let XOrth = m.XOrth;
    let FeaturesCS = features.center('column').scale('column');
    let labelsCS = group.center('column').scale('column');
    let Xres = FeaturesCS.clone().sub(XOrth);
    let plsCall = new NIPALS(Xres, { Y: labelsCS });
    let E = Xres.clone().sub(plsCall.t.clone().mmul(plsCall.p));

    let R2x = this.model.map((x) => x.R2x);
    let R2y = this.model.map((x) => x.R2y);

    this.output = { Q2y: Q2, // ok
      R2x, // ok
      R2y, // ok
      tPred: m.plsC.t,
      pPred: m.plsC.p,
      wPred: m.plsC.w,
      betasPred: m.plsC.betas,
      Qpc: m.plsC.q,
      tCV, // ok
      tOrthCV, // ok
      tOrth: m.tOrth,
      pOrth: m.pOrth,
      wOrth: m.wOrth,
      XOrth,
      Yres: m.plsC.yResidual,
      E };
  }

  /**
   * get access to all the computed elements
   * Mainly for debug and testing
   * @return {Object} output object
   */
  getResults() {
    return this.output;
  }

  getScores() {
    let scoresX = this.tCV.map((x) => x.to1DArray());
    let scoresY = this.tOrthCV.map((x) => x.to1DArray());
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
      tOrthCV: this.tOrthCV
    };
  }

  /**
   * Predict scores for new data
   * @param {Array} features a double array with X matrix
   * @param {Object} [options]
   * @param {Array} [options.trueLabel] an array with true values to compute confusion matrix
   * @param {Number} [options.nc] the number of components to be used
   * @return {Object} predictions
   */
  predict(features, options = {}) {
    var { trueLabels = [], nc = 1 } = options;
    let confusion = false;
    if (trueLabels.length > 0) {
      trueLabels = Matrix.from1DArray(150, 1, trueLabels);
      confusion = true;
    }

    // scaling the test dataset with respect to the train
    if (this.center) {
      features.center('column', { center: this.means });
      if (confusion) {
        trueLabels.center('column', { center: this.means });
      }
    }
    if (this.scale) {
      features.scale('column', { scale: this.stdevs });
      if (confusion) {
        trueLabels.scale('column', { center: this.means });
      }
    }

    let Eh = features;
    // removing the orthogonal components from PLS
    let tOrth;
    let wOrth;
    let pOrth;
    let yHat;
    let tPred;

    for (let idx = 0; idx < nc; idx++) {
      wOrth = this.model[idx].wOrth.transpose();
      pOrth = this.model[idx].pOrth;
      tOrth = Eh.clone().mmul(wOrth);
      Eh.sub(tOrth.clone().mmul(pOrth));
      // prediction
      tPred = Eh.clone().mmul(this.model[idx].plsC.w.transpose());
      // this should be summed over ncomp (pls_prediction.R line 23)
      yHat = tPred.clone().mmul(this.model[idx].plsC.betas);
    }
    let confusionMatrix = [];
    if (confusion) {
      confusionMatrix = ConfusionMatrix
        .fromLabels(trueLabels.to1DArray(), yHat.to1DArray());
    }
    return { tPred,
      tOrth,
      yHat,
      confusionMatrix };
  }

  _predictAll(features, labels, options = {}) {
    // cannot use the global this.center here
    // since it is used in the NC loop and
    // centering and scaling should only be
    // performed once
    const { center = true,
      scale = true } = options;

    if (center) {
      features.center('column');
      labels.center('column');
    }

    if (scale) {
      features.scale('column');
      labels.scale('column');
      // reevaluate tssy and tssx after scaling
      this.tssy = tss(labels);
      this.tssx = tss(features);
    }

    let oplsC = oplsNIPALS(features, labels);
    let plsC = new NIPALS(oplsC.filteredX, { Y: labels });

    let tPred = oplsC.filteredX.clone().mmul(plsC.w.transpose());
    let yHat = tPred.clone().mmul(plsC.betas);

    let rss = tss(labels.clone().sub(yHat));
    let R2y = 1 - (rss / this.tssy);

    let xEx = plsC.t.clone().mmul(plsC.p.clone());
    let rssx = tss(xEx);
    let R2x = (rssx / this.tssx);

    return { R2y,
      R2x,
      xRes: oplsC.filteredX,
      tOrth: oplsC.scoresXOrtho,
      pOrth: oplsC.loadingsXOrtho,
      wOrth: oplsC.weightsXOrtho,
      tPred: tPred,
      totalPred: yHat,
      XOrth: oplsC.scoresXOrtho.clone().mmul(oplsC.loadingsXOrtho),
      oplsC,
      plsC };
  }

  _getTrainTest(X, group, index) {
    let testFeatures = new Matrix(index.testIndex.length, X.columns);
    let testLabels = new Matrix(index.testIndex.length, 1);
    index.testIndex.forEach((el, idx) => {
      testFeatures.setRow(idx, X.getRow(el));
      testLabels.setRow(idx, group.getRow(el));
    });

    let trainFeatures = new Matrix(index.trainIndex.length, X.columns);
    let trainLabels = new Matrix(index.trainIndex.length, 1);
    index.trainIndex.forEach((el, idx) => {
      trainFeatures.setRow(idx, X.getRow(el));
      trainLabels.setRow(idx, group.getRow(el));
    });

    return ({
      trainFeatures,
      testFeatures,
      trainLabels,
      testLabels
    });
  }
}

