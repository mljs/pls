import { isAnyArray } from 'is-any-array';
import { ConfusionMatrix } from 'ml-confusion-matrix';
import { getFolds } from 'ml-cross-validation';
import { Matrix, NIPALS } from 'ml-matrix';
import { getAuc, getClasses, getRocCurve } from 'ml-roc-multiclass';

import { oplsNipals } from './oplsNipals.js';
import { getSafeStandardDeviations } from './util/getSafeStandardDeviations.js';
import { tss } from './util/tss.js';

/**
 * OPLS (orthogonal projections to latent structures).
 */
export class OPLS {
  /**
   * Creates a new OPLS model from features and labels.
   * @param {Array} data - matrix containing data (X).
   * @param {Array} labels - 1D Array containing metadata (Y). Numeric labels trigger regression, string labels trigger discriminant analysis.
   * @param {object} [options={}] - constructor options.
   * @param {boolean} [options.center=true] - should the data be centered (subtract the mean).
   * @param {boolean} [options.scale=true] - should the data be scaled (divide by the standard deviation).
   * @param {Array} [options.cvFolds=[]] - Allows to provide folds as array of objects with the arrays trainIndex and testIndex as properties.
   * @param {number} [options.nbFolds=7] - Allows to generate the defined number of folds with the training and test set chosen randomly from the data set.
   */
  constructor(data, labels, options = {}) {
    if (data === true) {
      const opls = options;
      this.labels = opls.labels;
      this.center = opls.center;
      this.scale = opls.scale;
      this.means = opls.means;
      this.meansY = opls.meansY;
      this.stdevs = opls.stdevs;
      this.stdevsY = opls.stdevsY;
      this.model = opls.model;
      this.predictiveScoresCV = opls.predictiveScoresCV;
      this.orthogonalScoresCV = opls.orthogonalScoresCV;
      this.yHatScoresCV = opls.yHatScoresCV;
      this.mode = opls.mode;
      this.output = opls.output;
      return;
    }

    const features = new Matrix(data);
    // set default values
    // cvFolds allows to define folds for testing purpose
    const { center = true, scale = true, cvFolds = [], nbFolds = 7 } = options;

    this.labels = labels;
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
      // constant columns (sd = 0) would divide by zero when scaling and produce
      // NaN; getSafeStandardDeviations replaces those zeros by 1 (check opls.R line 70).
      this.stdevs = getSafeStandardDeviations(features);
      this.stdevsY = getSafeStandardDeviations(group);
    } else {
      this.means = null;
    }

    const folds = cvFolds.length > 0 ? cvFolds : getFolds(labels, nbFolds);
    const Q2 = [];
    const aucResult = [];
    this.model = [];
    this.predictiveScoresCV = [];
    this.orthogonalScoresCV = [];
    this.yHatScoresCV = [];
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
      const predictiveScoresK = new Matrix(group.rows, 1);
      const orthogonalScoresK = new Matrix(group.rows, 1);
      oplsCV[nc] = [];
      for (let f = 0; f < folds.length; f++) {
        const trainTest = this._getTrainTest(features, group, folds[f]);
        const testFeatures = trainTest.testFeatures;
        const trainFeatures = trainTest.trainFeatures;
        const trainLabels = trainTest.trainLabels;
        // determine center and scale of training set
        const dataCenter = trainFeatures.mean('column');
        const dataSD = getSafeStandardDeviations(trainFeatures);

        // center and scale training set
        if (center) {
          trainFeatures.center('column');
          trainLabels.center('column');
        }

        if (scale) {
          // std computed after centering to stay bit-identical to the previous
          // internal scale('column'); sanitized to avoid dividing by a zero sd.
          trainFeatures.scale('column', {
            scale: getSafeStandardDeviations(trainFeatures),
          });
          trainLabels.scale('column');
        }
        // perform opls
        let oplsk;
        if (nc === 0) {
          oplsk = oplsNipals(trainFeatures, trainLabels);
        } else {
          oplsk = oplsNipals(oplsCV[nc - 1][f].filteredX, trainLabels);
        }

        // store model for next component
        oplsCV[nc][f] = oplsk;
        const plsCV = new NIPALS(oplsk.filteredX, { Y: trainLabels });

        // scaling the test dataset with respect to the train
        testFeatures.center('column', { center: dataCenter });
        testFeatures.scale('column', { scale: dataSD });

        const Eh = testFeatures;
        // removing the orthogonal components from PLS
        let scores;
        for (let idx = 0; idx < nc + 1; idx++) {
          scores = Eh.mmul(oplsCV[idx][f].weightsXOrtho.transpose()); // ok
          Eh.sub(scores.mmul(oplsCV[idx][f].loadingsXOrtho));
        }
        // prediction
        const predictiveComponents = Eh.mmul(plsCV.w.transpose());
        const yHatComponents = predictiveComponents
          .mmul(plsCV.betas)
          .mmul(plsCV.q.transpose()); // ok

        const yHat = new Matrix(yHatComponents.rows, 1);
        for (let i = 0; i < yHatComponents.rows; i++) {
          yHat.setRow(i, [yHatComponents.getRowVector(i).sum()]);
        }
        // adding all prediction from all folds
        for (let i = 0; i < folds[f].testIndex.length; i++) {
          yHatk.setRow(folds[f].testIndex[i], [yHat.get(i, 0)]);
          predictiveScoresK.setRow(folds[f].testIndex[i], [
            predictiveComponents.get(i, 0),
          ]);
          orthogonalScoresK.setRow(folds[f].testIndex[i], [scores.get(i, 0)]);
        }
      } // end of loop over folds
      this.predictiveScoresCV.push(predictiveScoresK);
      this.orthogonalScoresCV.push(orthogonalScoresK);
      this.yHatScoresCV.push(yHatk);

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
        overfitted = value - listOfValues[nc - 1] < 0.05;
      }
      this.model.push(modelNC);
      // store the model for each component
      nc++;
      // console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
    } while (!overfitted); // end of loop over nc
    // store scores from CV
    const predictiveScoresCV = this.predictiveScoresCV;
    const orthogonalScoresCV = this.orthogonalScoresCV;
    const yHatScoresCV = this.yHatScoresCV;
    const m = this.model[nc - 1];
    const orthogonalData = new Matrix(features.rows, features.columns);
    const orthogonalScores = new Matrix(features.rows, nc - 1);
    const orthogonalLoadings = new Matrix(nc - 1, features.columns);
    const orthogonalWeights = new Matrix(nc - 1, features.columns);
    for (let i = 0; i < this.model.length - 1; i++) {
      orthogonalData.add(this.model[i].XOrth);
      orthogonalScores.setSubMatrix(this.model[i].orthogonalScores, 0, i);
      orthogonalLoadings.setSubMatrix(this.model[i].orthogonalLoadings, i, 0);
      orthogonalWeights.setSubMatrix(this.model[i].orthogonalWeights, i, 0);
    }

    const FeaturesCS = features.center('column');
    FeaturesCS.scale('column', {
      scale: getSafeStandardDeviations(FeaturesCS),
    });
    let labelsCS;
    if (this.mode === 'regression') {
      labelsCS = group.clone().center('column').scale('column');
    } else {
      labelsCS = group;
    }

    const orthogonalizedData = FeaturesCS.clone().sub(orthogonalData);
    const plsCall = new NIPALS(orthogonalizedData, { Y: labelsCS });
    const residualData = orthogonalizedData
      .clone()
      .sub(plsCall.t.mmul(plsCall.p));
    const R2x = this.model.map((x) => x.R2x);
    const R2y = this.model.map((x) => x.R2y);

    this.output = {
      Q2y: Q2,
      auc: aucResult,
      R2x,
      R2y,
      predictiveComponents: plsCall.t,
      predictiveLoadings: plsCall.p,
      predictiveWeights: plsCall.w,
      betas: plsCall.betas,
      Qpc: plsCall.q,
      predictiveScoresCV,
      orthogonalScoresCV,
      yHatScoresCV,
      oplsCV,
      orthogonalScores,
      orthogonalLoadings,
      orthogonalWeights,
      Xorth: orthogonalData,
      yHat: m.totalPred,
      Yres: m.plsC.yResidual,
      residualData,
      folds,
    };
  }

  /**
   * get access to all the computed elements
   * Mainly for debug and testing
   * @returns {object} output object
   */
  getLogs() {
    return this.output;
  }

  /**
   * Returns the cross-validated predictive and orthogonal scores.
   * @returns {{scoresX: Array<Array<number>>, scoresY: Array<Array<number>>}} the predictive (`scoresX`) and orthogonal (`scoresY`) cross-validated scores.
   */
  getScores() {
    const scoresX = this.predictiveScoresCV.map((x) => x.to1DArray());
    const scoresY = this.orthogonalScoresCV.map((x) => x.to1DArray());
    return { scoresX, scoresY };
  }

  /**
   * Load an OPLS model from JSON
   * @param {object} model - the serialized model to load.
   * @returns {OPLS} the loaded OPLS model.
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
   * @returns {object} model
   */
  toJSON() {
    return {
      name: 'OPLS',
      labels: this.labels,
      center: this.center,
      scale: this.scale,
      means: this.means,
      stdevs: this.stdevs,
      model: this.model,
      mode: this.mode,
      output: this.output,
      predictiveScoresCV: this.predictiveScoresCV,
      orthogonalScoresCV: this.orthogonalScoresCV,
      yHatScoresCV: this.yHatScoresCV,
    };
  }

  /**
   * Predicts the class of each row of new data (discriminant analysis mode).
   * @param {Matrix} features - a matrix containing new data.
   * @param {object} [options={}] - prediction options.
   * @param {Array} [options.trueLabels] - an array with true values to compute confusion matrix.
   * @param {boolean} [options.center=this.center] - should the data be centered before prediction.
   * @param {boolean} [options.scale=this.scale] - should the data be scaled before prediction.
   * @returns {Array} the predicted class name for each row of `features`.
   */
  predictCategory(features, options = {}) {
    const {
      trueLabels = [],
      center = this.center,
      scale = this.scale,
    } = options;
    if (isAnyArray(features)) {
      if (features[0].length === undefined) {
        features = Matrix.from1DArray(1, features.length, features);
      } else {
        features = Matrix.checkMatrix(features);
      }
    }
    const prediction = this.predict(features, { trueLabels, center, scale });
    const predictiveComponents = Matrix.checkMatrix(
      this.output.predictiveComponents,
    ).to1DArray();
    const newTPred = Matrix.checkMatrix(
      prediction.predictiveComponents,
    ).to1DArray();
    const categories = getClasses(this.labels);
    const classes = this.labels.slice();
    const result = [];
    for (const pred of newTPred) {
      let item;
      let auc = 0;
      for (const category of categories) {
        const testTPred = predictiveComponents.slice();
        testTPred.push(pred);
        const testClasses = classes.slice();
        testClasses.push(category.name);
        const rocCurve = getRocCurve(testClasses, testTPred);
        const areaUnderCurve = getAuc(rocCurve);
        if (auc < areaUnderCurve) {
          item = category.name;
          auc = areaUnderCurve;
        }
      }
      result.push(item);
    }
    return result;
  }

  /* eslint-disable-next-line jsdoc/require-returns-check -- always returns for the supported modes (regression / discriminantAnalysis) */
  /**
   * Predict scores for new data
   * @param {Matrix} features - a matrix containing new data
   * @param {object} [options={}] - prediction options.
   * @param {Array} [options.trueLabels] - an array with true values to compute confusion matrix.
   * @param {boolean} [options.center=this.center] - should the data be centered before prediction.
   * @param {boolean} [options.scale=this.scale] - should the data be scaled before prediction.
   * @returns {object} - predictions
   */
  predict(features, options = {}) {
    const {
      trueLabels = [],
      center = this.center,
      scale = this.scale,
    } = options;

    let labels;
    if (typeof trueLabels[0] === 'number') {
      labels = Matrix.from1DArray(trueLabels.length, 1, trueLabels);
    } else if (typeof trueLabels[0] === 'string') {
      labels = Matrix.checkMatrix(createDummyY(trueLabels)).transpose();
    }

    features = new Matrix(features);

    // scaling the test dataset with respect to the train
    if (center) {
      features.center('column', { center: this.means });
      if (labels?.rows > 0) {
        labels.center('column', { center: this.meansY });
      }
    }
    if (scale) {
      features.scale('column', { scale: this.stdevs });
      if (labels?.rows > 0) {
        labels.scale('column', { scale: this.stdevsY });
      }
    }

    const nc =
      this.mode === 'regression'
        ? this.model[0].Q2y.length
        : this.model[0].auc.length - 1;

    const Eh = features.clone();
    // removing the orthogonal components from PLS
    let orthogonalScores;
    let orthogonalWeights;
    let orthogonalLoadings;
    let totalPred;
    let predictiveComponents;
    for (let idx = 0; idx < nc; idx++) {
      const model = this.model[idx];
      orthogonalWeights = Matrix.checkMatrix(
        model.orthogonalWeights,
      ).transpose();
      orthogonalLoadings = Matrix.checkMatrix(model.orthogonalLoadings);
      orthogonalScores = Eh.mmul(orthogonalWeights);
      Eh.sub(orthogonalScores.mmul(orthogonalLoadings));
      // prediction
      predictiveComponents = Eh.mmul(
        Matrix.checkMatrix(model.plsC.w).transpose(),
      );
      const components = predictiveComponents
        .mmul(model.plsC.betas)
        .mmul(Matrix.checkMatrix(model.plsC.q).transpose());
      totalPred = new Matrix(components.rows, 1);
      for (let i = 0; i < components.rows; i++) {
        totalPred.setRow(i, [components.getRowVector(i).sum()]);
      }
    }

    if (labels?.rows > 0) {
      if (this.mode === 'regression') {
        const tssy = tss(labels);
        const press = tss(labels.clone().sub(totalPred));
        const Q2y = 1 - press / tssy;

        return { predictiveComponents, orthogonalScores, yHat: totalPred, Q2y };
      } else if (this.mode === 'discriminantAnalysis') {
        const confusionMatrix = ConfusionMatrix.fromLabels(
          trueLabels,
          totalPred.to1DArray(),
        );
        const rocCurve = getRocCurve(trueLabels, totalPred.to1DArray());
        const auc = getAuc(rocCurve);
        return {
          predictiveComponents,
          orthogonalScores,
          yHat: totalPred,
          confusionMatrix,
          auc,
        };
      }
    } else {
      return { predictiveComponents, orthogonalScores, yHat: totalPred };
    }
  }

  /**
   * Fits a one-component OPLS model on the full data set and computes its R2 statistics.
   * @private
   * @param {Matrix} data - the feature matrix (X).
   * @param {Matrix} categories - the label matrix (Y).
   * @param {object} [options={}] - prediction options.
   * @param {boolean} [options.center=true] - should the data be centered.
   * @param {boolean} [options.scale=true] - should the data be scaled.
   * @returns {object} the fitted component with its scores, loadings, weights and R2x/R2y statistics.
   */
  _predictAll(data, categories, options = {}) {
    // cannot use the global this.center here
    // since it is used in the NC loop and
    // centering and scaling should only be
    // performed once
    const { center = true, scale = true } = options;
    const features = data.clone();
    const labels = categories.clone();

    if (center) {
      const means = features.mean('column');
      features.center('column', { center: means });
      labels.center('column');
    }
    if (scale) {
      const stdevs = getSafeStandardDeviations(features);
      features.scale('column', { scale: stdevs });
      labels.scale('column');
      // reevaluate tssy and tssx after scaling
      // must be global because re-used for next nc iteration
      // tssx is only evaluate the first time
      this.tssy = tss(labels);
      this.tssx = tss(features);
    }
    const oplsC = oplsNipals(features, labels);
    const plsC = new NIPALS(oplsC.filteredX, { Y: labels });
    const predictiveComponents = plsC.t.clone();
    // const yHat = tPred.mmul(plsC.betas).mmul(plsC.q.transpose()); // ok
    const yHatComponents = predictiveComponents
      .mmul(plsC.betas)
      .mmul(plsC.q.transpose()); // ok
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
      orthogonalScores: oplsC.scoresXOrtho,
      orthogonalLoadings: oplsC.loadingsXOrtho,
      orthogonalWeights: oplsC.weightsXOrtho,
      predictiveComponents,
      totalPred: yHat,
      XOrth: oplsC.scoresXOrtho.mmul(oplsC.loadingsXOrtho),
      oplsC,
      plsC,
    };
  }
  /**
   * Splits the data into train and test sets for the given fold.
   * @private
   * @param {Matrix} X - dataset matrix object.
   * @param {Matrix} group - labels matrix object.
   * @param {object} index - train and test index (output from getFold()).
   * @returns {object} train and test features and labels.
   */
  _getTrainTest(X, group, index) {
    const testFeatures = new Matrix(index.testIndex.length, X.columns);
    const testLabels = new Matrix(index.testIndex.length, group.columns);
    for (const [idx, el] of index.testIndex.entries()) {
      testFeatures.setRow(idx, X.getRow(el));
      testLabels.setRow(idx, group.getRow(el));
    }

    const trainFeatures = new Matrix(index.trainIndex.length, X.columns);
    const trainLabels = new Matrix(index.trainIndex.length, group.columns);
    for (const [idx, el] of index.trainIndex.entries()) {
      trainFeatures.setRow(idx, X.getRow(el));
      trainLabels.setRow(idx, group.getRow(el));
    }

    return {
      trainFeatures,
      testFeatures,
      trainLabels,
      testLabels,
    };
  }
}

/**
 * Builds a dummy (indicator) Y matrix from an array of class labels.
 * For more than two classes each row encodes one class as `1` and the others as `-1`;
 * for two classes a single row of `2`/`1` values is returned.
 * @private
 * @param {Array} array - the array of class labels.
 * @returns {Array<Array<number>>} the dummy Y matrix.
 */
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
