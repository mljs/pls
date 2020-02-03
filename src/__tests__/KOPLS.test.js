import Matrix from 'ml-matrix';
import Kernel from 'ml-kernel';
import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';

import { KOPLS } from '../KOPLS.js';

expect.extend({ toBeDeepCloseTo });

describe('K-OPLS', () => {
  let Xtest = new Matrix(require('../../data/Xtest.json'));
  let Xtrain = new Matrix(require('../../data/Xtrain.json'));
  let Ytest = require('../../data/Ytest.json');
  let Ytrain = new Matrix(require('../../data/Ytrain.json'));
  let Tp = new Matrix(require('../../data/tp.json'));
  let to = new Matrix(require('../../data/to.json'));

  let kernel = new Kernel('gaussian', {
    sigma: 25,
  });

  let cls = new KOPLS({
    orthogonalComponents: 10,
    predictiveComponents: 1,
    kernel: kernel,
  });

  cls.train(Xtrain, Ytrain);

  it('K-OPLS test with main features', () => {
    let { prediction, predScoreMat, predYOrthVectors } = cls.predict(Xtest);

    for (let i = 0; i < predScoreMat.length; ++i) {
      for (let j = 0; j < predScoreMat[i].length; ++j) {
        expect(predScoreMat[i][j][0]).toBeCloseTo(Tp[i][j], 2);
      }
    }

    for (let i = 0; i < predYOrthVectors.length; ++i) {
      for (let j = 0; j < predYOrthVectors[i].length; ++j) {
        expect(predYOrthVectors[i][j][0]).toBeCloseTo(to[i][j], 2);
      }
    }

    expect(prediction.to2DArray()).toBeDeepCloseTo(Ytest, 3);
  });

  it('Load and save', () => {
    let model = KOPLS.load(JSON.parse(JSON.stringify(cls)), kernel);
    let output = model.predict(Xtest).prediction;

    expect(output.to2DArray()).toBeDeepCloseTo(Ytest, 3);
  });

  it('Test with real dataset', () => {
    Xtest = new Matrix(require('../../data/Xtest1.json'));
    Xtrain = new Matrix(require('../../data/Xtrain1.json'));
    Ytest = require('../../data/Ytest1.json');
    Ytrain = new Matrix(require('../../data/Ytrain1.json'));

    cls = new KOPLS({
      orthogonalComponents: 10,
      predictiveComponents: 2,
      kernel: kernel,
    });

    cls.train(Xtrain, Ytrain);
    let output = cls.predict(Xtest).prediction;
    expect(output.to2DArray()).toBeDeepCloseTo(Ytest, 1);
  });
});
