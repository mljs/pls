import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { ConfusionMatrix } from 'ml-confusion-matrix';
import { sampleAClass } from 'ml-cross-validation';
import {
  getNumbers,
  getClasses,
  getCrossValidationSets,
} from 'ml-dataset-iris';
import { METADATA } from 'ml-dataset-metadata';
import { Matrix, NIPALS } from 'ml-matrix';

import { OPLS } from '../OPLS.js';
import { oplsNipals } from '../oplsNipals.js';
import { tss } from '../util/tss.js';

expect.extend({ toBeDeepCloseTo });

const iris = getNumbers();
const metadata = getClasses();
const folds = getCrossValidationSets(7, { idx: 0, by: 'folds' });
const newM = new METADATA([metadata], { headers: ['iris'] });

describe('centering and scaling X and Y', () => {
  let x = new Matrix(iris);

  it('test that iris X scaling is similar to scaling in R', () => {
    x = x.center('column').scale('column');
    expect(x.get(0, 0)).toBeCloseTo(-0.8976739, 6); // ok
  });

  it('test that iris Y scaling is similar to scaling in R', () => {
    let y = newM.get('iris', { format: 'matrix' }).values;
    y = y.center('column').scale('column');
    expect(y.get(0, 0)).toBeCloseTo(-1.220656, 6); // ok
  });
});

describe('OPLS nipals components', () => {
  it('test first opls loop nComp = 1 and fold = 1', () => {
    let rawData = iris;
    let testRawData = rawData.filter((el, idx) => !folds[0].includes(idx));
    rawData = rawData.filter((el, idx) => folds[0].includes(idx));
    const x = new Matrix(rawData);
    const center = x.mean('column');
    const sd = x.standardDeviation('column');
    x.center('column').scale('column');
    const toExclude = getCrossValidationSets(7, { idx: 0, by: 'trainTest' })[0]
      .testIndex;
    const met = getClasses();
    let y = new METADATA([met], { headers: ['iris'] });
    y.remove(toExclude, 'row');
    y = y.get('iris', { format: 'matrix' }).values;
    y = y.center('column').scale('column');
    const opls = oplsNipals(x, y);

    expect(folds[0].reduce((a, c) => a + c)).toBe(9343); // ok
    expect(y.rows).toBe(127); // ok
    expect(y.get(0, 0)).toBeCloseTo(-1.22030407, 6); // ok
    expect(y.get(126, 0)).toBeCloseTo(1.2593538, 6); // ok
    expect(x.get(0, 0)).toBeCloseTo(-0.86171924, 6); // ok

    expect(opls.filteredX.get(0, 0)).toBeCloseTo(-1.055718013, 8); // ok
    expect(opls.filteredX.get(3, 3)).toBeCloseTo(-1.316108827, 8);
    expect(opls.filteredX.get(126, 2)).toBeCloseTo(0.949878685, 8);

    expect(opls.scoresXpred.get(0, 0)).toBeCloseTo(-2.258296156, 8);
    expect(opls.scoresXpred.get(126, 0)).toBeCloseTo(1.003592216, 8);
    expect(opls.scoresXpred.sum()).toBeCloseTo(-3.468059e-14, 8);

    expect(opls.loadingsXpred.get(0, 0)).toBeCloseTo(0.525843, 6);
    expect(opls.loadingsXpred.get(0, 1)).toBeCloseTo(-0.2431478, 6);
    expect(opls.loadingsXpred.get(0, 3)).toBeCloseTo(0.5698576, 6);

    expect(opls.weightsXPred.get(0, 0)).toBeCloseTo(0.4852953, 6);
    expect(opls.weightsXPred.get(1, 0)).toBeCloseTo(-0.2406125, 6);

    expect(opls.weightsXOrtho.get(0, 0)).toBeCloseTo(0.8165095, 6);
    expect(opls.weightsXOrtho.get(0, 2)).toBeCloseTo(-0.1230943, 6);

    expect(opls.loadingsY.get(0, 0)).toBeCloseTo(0.5561059, 6);
    expect(opls.scoresXOrtho.get(0, 0)).toBeCloseTo(0.144701329, 8);
    expect(opls.loadingsXOrtho.get(0, 0)).toBeCloseTo(1.340684);

    const xResidual = opls.filteredX;
    const plsComp = new NIPALS(xResidual, { Y: y });

    expect(plsComp.xResidual.get(0, 0)).toBeCloseTo(0.0746564622, 2);
    expect(plsComp.xResidual.get(3, 3)).toBeCloseTo(0.1006373429, 2);
    expect(plsComp.xResidual.get(126, 2)).toBeCloseTo(0.1542974407, 2);

    expect(plsComp.yResidual.get(0, 0)).toBeCloseTo(0.148422167, 8);
    expect(plsComp.yResidual.get(3, 0)).toBeCloseTo(0.166480767, 8);
    expect(plsComp.yResidual.get(126, 0)).toBeCloseTo(0.477856213, 8);

    expect(plsComp.t.get(0, 0)).toBeCloseTo(-2.36832783, 8);
    expect(plsComp.t.get(4, 0)).toBeCloseTo(-2.21912839, 8);
    expect(plsComp.t.get(125, 0)).toBeCloseTo(1.90700548, 8);

    expect(plsComp.p.get(0, 0)).toBeCloseTo(0.477288, 3);
    expect(plsComp.p.get(0, 3)).toBeCloseTo(0.5904155, 2);
    expect(plsComp.w.get(0, 0)).toBeCloseTo(0.4852953, 6);
    expect(plsComp.w.get(0, 2)).toBeCloseTo(0.5910032, 6);
    expect(plsComp.betas.get(0, 0)).toBeCloseTo(0.5779294, 3);

    // scaling of the test dataset with respect to the train
    testRawData = testRawData.map((d) => d.slice(0, 4));
    const testx = new Matrix(23, 4);
    testRawData.forEach((el, i) => testx.setRow(i, testRawData[i]));
    testx.center('column', { center });
    testx.scale('column', { scale: sd });

    expect(testx.get(0, 0)).toBeCloseTo(-1.10145274, 6);
    expect(testx.get(3, 3)).toBeCloseTo(-1.1829865, 6);

    let Eh = testx;
    // removing the orthogonal components from PLS
    const scores = Eh.mmul(opls.weightsXOrtho.transpose());
    Eh = Eh.clone().sub(scores.mmul(opls.loadingsXOrtho));

    expect(scores.get(0, 0)).toBeCloseTo(0.009042318, 6);
    expect(scores.get(4, 0)).toBeCloseTo(-0.103439622, 6);
    expect(Eh.get(0, 0)).toBeCloseTo(-1.11357563, 6);
    expect(Eh.get(22, 1)).toBeCloseTo(-1.22642102, 6);

    const predictiveComponents = Eh.mmul(plsComp.w.transpose());
    expect(predictiveComponents.get(0, 0)).toBeCloseTo(-2.0983292, 6);
    expect(predictiveComponents.get(7, 0)).toBeCloseTo(-2.4324436, 6);

    const yHat = predictiveComponents.mmul(plsComp.betas);
    expect(yHat.get(0, 0)).toBeCloseTo(-1.21268611, 6);
    expect(yHat.get(7, 0)).toBeCloseTo(-1.40578061, 6);
  });

  it('test first opls nipals loop nComp = 1 with all folds', () => {
    const cvPreds = new Matrix(150, 1);
    const cvScoresO = new Matrix(150, 1);
    const cvScoresP = new Matrix(150, 1);

    for (let fold of folds) {
      let rawData = iris;
      let testRawData = rawData.filter((el, idx) => !fold.includes(idx));
      rawData = rawData.filter((el, idx) => fold.includes(idx));
      const x = new Matrix(rawData);
      const center = x.mean('column');
      const sd = x.standardDeviation('column');
      x.center('column').scale('column');

      let y = metadata.filter((el, idx) => fold.includes(idx));
      const M = new METADATA([y], { headers: ['iris'] });
      y = M.get('iris', { format: 'matrix' }).values;
      y = y.center('column').scale('column');

      const opls = oplsNipals(x, y);
      const xResidual = opls.filteredX;
      const plsComp = new NIPALS(xResidual, { Y: y });

      testRawData = testRawData.map((d) => d.slice(0, 4));
      const testx = new Matrix(testRawData.length, 4);
      testRawData.forEach((el, i) => testx.setRow(i, testRawData[i]));
      testx.center('column', { center });
      testx.scale('column', { scale: sd });

      const Eh = testx;
      // removing the orthogonal components from PLS
      const scores = Eh.mmul(opls.weightsXOrtho.transpose());
      Eh.sub(scores.mmul(opls.loadingsXOrtho));

      const predictiveComponents = Eh.mmul(plsComp.w.transpose());
      const yHat = predictiveComponents.mmul(plsComp.betas);
      let testCv = [];
      for (let j = 0; j < 150; j++) {
        testCv.push(j);
      }

      testCv = testCv.filter((el, idx) => !fold.includes(idx));
      testCv.forEach((el, idx) => cvPreds.setRow(el, [yHat.get(idx, 0)]));
      testCv.forEach((el, idx) => cvScoresO.setRow(el, [scores.get(idx, 0)]));
      testCv.forEach((el, idx) =>
        cvScoresP.setRow(el, [predictiveComponents.get(idx, 0)]),
      );
    }

    const y = newM.get('iris', { format: 'matrix' }).values;
    y.center('column').scale('column');
    const tssy = tss(y);
    const press = tss(y.clone().sub(cvPreds));

    expect(cvPreds.get(0, 0)).toBeCloseTo(-1.42935863, 6);
    expect(cvPreds.get(3, 0)).toBeCloseTo(-1.16151882, 6);
    expect(cvScoresO.get(0, 0)).toBeCloseTo(0.078273936, 6);
    expect(cvScoresP.get(0, 0)).toBeCloseTo(-2.48581401, 6);
    expect(press).toBeCloseTo(11.78251, 5);
    expect(tssy).toBeCloseTo(149, 10);
    expect(1 - press / tssy).toBeCloseTo(0.9209228, 6);
  });
});

describe('OPLS utility functions', () => {
  it('test tss', () => {
    let y = newM.get('iris', { format: 'matrix' }).values;
    y = y.center('column').scale('column');

    let x = new Matrix(iris);
    x = x.center('column').scale('column');
    expect(x.get(0, 0)).toBeCloseTo(-0.8976739, 6);
    expect(y.get(0, 0)).toBeCloseTo(-1.220656, 6);
    expect(tss(x)).toBeCloseTo(596, 6);
    expect(tss(y)).toBe(149);
  });

  it('test total prediction', () => {
    let y = newM.get('iris', { format: 'matrix' }).values;
    y = y.center('column').scale('column');

    let x = new Matrix(iris);
    x = x.center('column').scale('column');

    const res = oplsNipals(x, y);
    const xRes = res.filteredX;
    expect(xRes.get(0, 0)).toBeCloseTo(-0.99598366, 6);
    expect(res.scoresXOrtho.get(0, 0)).toBeCloseTo(0.074537852, 6);

    const plsComp = new NIPALS(xRes, { Y: y });

    expect(plsComp.t.get(0, 0)).toBeCloseTo(-2.32295367, 6);
    expect(plsComp.xResidual.get(0, 1)).toBeCloseTo(0.340980373, 3);
    expect(plsComp.yResidual.get(0, 0)).toBeCloseTo(0.1143555571, 6);
    expect(plsComp.p.get(0, 1)).toBeCloseTo(-0.2864595, 3);
    expect(plsComp.w.get(0, 0)).toBeCloseTo(0.484385, 6);

    const predictiveComponents = xRes.mmul(plsComp.w.transpose());
    const yHat = predictiveComponents.mmul(plsComp.betas);

    expect(predictiveComponents.get(0, 0)).toBeCloseTo(-2.32295367, 6);
    expect(yHat.get(0, 0)).toBeCloseTo(-1.33501112, 6);

    const tssy = tss(y);
    let rss = y.clone().sub(yHat);
    rss = rss.clone().mul(rss).sum();
    const R2y = 1 - rss / tssy;

    expect(tssy).toBeCloseTo(149, 6); // ok
    expect(rss).toBeCloseTo(10.65667, 5);
    expect(R2y).toBeCloseTo(0.9284787, 6);

    const xEx = plsComp.t.mmul(plsComp.p);
    const rssx = tss(xEx);
    const tssx = tss(x);
    const R2x = rssx / tssx;

    expect(rssx).toBeCloseTo(419.0932, 0);
    expect(tssx).toBeCloseTo(596, 6); // ok
    expect(R2x).toBeCloseTo(0.7031765, 2);
  });

  it('test OPLS sampleAClass', () => {
    const c = sampleAClass(metadata, 0.1).trainIndex;
    const d = [];
    c.forEach((el) => d.push(metadata[el]));
    let counts = {};
    d.forEach((x) => {
      counts[x] = (counts[x] || 0) + 1;
    });

    expect(sampleAClass(metadata, 0.1).trainIndex).toHaveLength(15);
    expect(sampleAClass(metadata, 0.1).testIndex).toHaveLength(135);
    expect(sampleAClass(metadata, 0.1).mask).toHaveLength(150);
    expect(JSON.stringify(counts)).toBe(
      JSON.stringify({ setosa: 5, versicolor: 5, virginica: 5 }),
    );
  });

  it('test OPLS sampleAClass 2', () => {
    const c = sampleAClass(metadata, 0.1).mask;
    const train = metadata.filter((f, idx) => c[idx]);
    const test = metadata.filter((f, idx) => !c[idx]);

    expect(train).toHaveLength(15);
    expect(test).toHaveLength(135);
    expect(sampleAClass(metadata, 0.1).mask).toHaveLength(150);
  });

  it('test OPLS dataArray', () => {
    const x = new Matrix(iris);
    const trainTestLabels = require('../../data/trainTestLabels.json');
    const M = new METADATA([metadata], { headers: ['iris'] });
    const labels = M.get('iris', { format: 'factor' }).values;
    const model = new OPLS(x, labels, { cvFolds: trainTestLabels });

    expect(model.model).toHaveLength(2);
  });
});

describe('OPLS', () => {
  it('test OPLS with 7 folds', () => {
    const x = new Matrix(iris);
    const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
    const labels = newM.get('iris', { format: 'factor' }).values;
    const model = new OPLS(x, labels, { cvFolds });

    expect(model.predictiveScoresCV[0].get(0, 0)).toBeCloseTo(-2.48581401, 6);
    expect(model.orthogonalScoresCV[0].get(0, 0)).toBeCloseTo(0.078273936, 6);
    expect(model.orthogonalScoresCV[1].get(0, 0)).toBeCloseTo(-0.439656132, 6);
    expect(model.predictiveScoresCV[1].get(0, 0)).toBeCloseTo(-2.453147, 6);
    expect(model.getLogs().Q2y[0]).toBeCloseTo(0.9209228, 6);
    expect(model.getLogs().Q2y[1]).toBeCloseTo(0.9263751, 6);
    expect(model.getLogs().R2y[0]).toBeCloseTo(0.9284787, 6);
    expect(model.getLogs().R2y[1]).toBeCloseTo(0.9301693, 6);
    expect(model.getLogs().R2x[0]).toBeCloseTo(0.7031765, 3);
    expect(model.getLogs().R2x[1]).toBeCloseTo(0.7015103, 3);
    expect(model.model[1].predictiveComponents.get(0, 0)).toBeCloseTo(
      -2.290801,
      6,
    );
    expect(model.model[0].predictiveComponents.get(0, 0)).toBeCloseTo(
      -2.32295367,
      6,
    );
    expect(model.model[0].orthogonalScores.get(0, 0)).toBeCloseTo(
      0.074537852,
      6,
    );
    expect(model.model[1].orthogonalScores.get(0, 0)).toBeCloseTo(
      -0.416881408,
      6,
    );
    expect(model.model[0].orthogonalScores.get(149, 0)).toBeCloseTo(
      -0.486465664,
      6,
    );
    expect(model.model[0].orthogonalLoadings.get(0, 0)).toBeCloseTo(
      1.318924,
      6,
    );
    expect(model.model[1].orthogonalLoadings.get(0, 0)).toBeCloseTo(
      -0.2600433,
      6,
    );
    expect(model.model[0].orthogonalWeights.get(0, 0)).toBeCloseTo(
      0.7888785,
      6,
    );
    expect(model.model[1].orthogonalWeights.get(0, 0)).toBeCloseTo(
      -0.2831915,
      6,
    );
    expect(model.model[0].plsC.betas.get(0, 0)).toBeCloseTo(0.5747042, 6);
    expect(model.model[1].plsC.betas.get(0, 0)).toBeCloseTo(0.5758896, 3);
    expect(model.model[0].plsC.q.get(0, 0)).toBeCloseTo(1, 6);
    expect(model.model[1].plsC.q.get(0, 0)).toBeCloseTo(1, 6);
    expect(model.model[0].plsC.p.get(0, 0)).toBeCloseTo(0.4777117, 3);
    expect(model.model[1].plsC.p.get(0, 0)).toBeCloseTo(0.484385, 3);
    expect(model.model[0].plsC.w.get(0, 0)).toBeCloseTo(0.484385, 6);
    expect(model.model[1].plsC.w.get(0, 0)).toBeCloseTo(0.484385, 6);
    expect(model.model[0].plsC.t.get(0, 0)).toBeCloseTo(-2.32295367, 6);
    expect(model.model[1].plsC.t.get(0, 0)).toBeCloseTo(-2.290801, 6);
    expect(model.model[0].plsC.yResidual.get(0, 0)).toBeCloseTo(
      0.1143555571,
      6,
    );
    expect(model.model[1].plsC.yResidual.get(0, 0)).toBeCloseTo(0.09827427, 6);
    expect(model.model[0].plsC.xResidual.get(0, 0)).toBeCloseTo(0.113718555, 3);
    expect(model.model[1].plsC.xResidual.get(0, 0)).toBeCloseTo(0.006007284, 3);
    expect(model.model[0].XOrth.get(0, 0)).toBeCloseTo(0.09830979, 6);
    expect(model.model[0].totalPred.get(0, 0)).toBeCloseTo(-1.33501112, 6);
    expect(model.model[0].oplsC.filteredX.get(0, 0)).toBeCloseTo(
      -0.99598366,
      6,
    );
    expect(model.model[0].oplsC.scoresXOrtho.get(0, 0)).toBeCloseTo(
      0.074537852,
      6,
    );
  });
});

describe('confusion matrix', () => {
  const trueLabels = [1];
  const predictedLabels = [1];
  const CM2 = ConfusionMatrix.fromLabels(trueLabels, predictedLabels);

  it('test confusion matrix works even with length 1', () => {
    expect(CM2.getAccuracy()).toBe(1);
  });
});

describe('import / export model', () => {
  const x = new Matrix(iris);
  const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
  const labels = newM.get('iris', { format: 'factor' }).values;
  const model = new OPLS(x, labels, { cvFolds });
  const exportedModel = JSON.stringify(model.toJSON());
  const newModel = OPLS.load(JSON.parse(exportedModel));
  const test = [
    [5.1, 3.5, 1.4, 0.2],
    [7, 3.2, 4.7, 1.4],
    [6.3, 3.3, 6, 2.5],
  ];
  it('test export', () => {
    expect(JSON.parse(exportedModel).name).toBe('OPLS');
  });
  it('test import', () => {
    expect(model.predict(test)).toStrictEqual(newModel.predict(test));
  });
  it('test category prediction', () => {
    expect(model.predictCategory(test)).toStrictEqual([0, 1, 2]);
    expect(newModel.predictCategory(test)).toStrictEqual([0, 1, 2]);
  });
});

describe('prediction', () => {
  const x = new Matrix(iris);
  const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
  const labels = newM.get('iris', { format: 'factor' }).values;
  const model = new OPLS(x, labels, { cvFolds });
  const prediction = model.predict(x, { trueLabels: labels });

  it('test prediction length', () => {
    expect(prediction.predictiveComponents.rows).toBe(150);
  });

  it('test prediction Q2y', () => {
    expect(prediction.Q2y).toBeCloseTo(0.93016, 4);
  });

  it('test prediction yHat vector', () => {
    expect(prediction.yHat.to1DArray()).toBeDeepCloseTo(
      model.getLogs().yHat.to1DArray(),
      5,
    );
  });
});

describe('prediction with metadata', () => {
  const x = new Matrix(iris);
  const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
  const model = new OPLS(x, metadata, { cvFolds });
  const prediction = model.predict(x, { trueLabels: metadata });

  it('test prediction length', () => {
    expect(prediction.predictiveComponents.rows).toBe(150);
  });

  it('test prediction predictiveComponents result', () => {
    expect(
      prediction.predictiveComponents.to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [2.36197253, 2.03663185, 2.18730941, 2.04038657, 2.41783405],
      8,
    );
  });

  it('test prediction orthogonalScores result', () => {
    expect(prediction.orthogonalScores.to1DArray().slice(0, 5)).toBeDeepCloseTo(
      [-0.120705304, 0.001762035, 0.221496966, 0.323480471, -0.006342759],
      8,
    );
  });

  it('test prediction predictiveComponents vector', () => {
    expect(prediction.predictiveComponents.to1DArray()).toBeDeepCloseTo(
      model.getLogs().predictiveComponents.to1DArray(),
      8,
    );
  });

  it('test prediction orthogonalScores vector', () => {
    expect(prediction.orthogonalScores.to1DArray()).toBeDeepCloseTo(
      model.getLogs().orthogonalScores.to1DArray(),
      8,
    );
  });

  it('test prediction yHat vector', () => {
    expect(prediction.yHat.to1DArray()).toBeDeepCloseTo(
      model.getLogs().yHat.to1DArray(),
      8,
    );
  });
});

describe('prediction without labels', () => {
  const x = new Matrix(iris);
  const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
  const labels = newM.get('iris', { format: 'factor' }).values;
  const model = new OPLS(x, labels, { cvFolds });
  const prediction = model.predict(x);

  it('test prediction length', () => {
    expect(prediction.predictiveComponents.rows).toBe(150);
  });

  it('test prediction yHat', () => {
    expect(prediction.yHat.get(0, 0)).toBeCloseTo(
      model.getLogs().yHat.get(0, 0),
      5,
    );
  });

  it('test prediction yHat vector', () => {
    expect(prediction.yHat.to1DArray()).toBeDeepCloseTo(
      model.getLogs().yHat.to1DArray(),
      5,
    );
  });
});
