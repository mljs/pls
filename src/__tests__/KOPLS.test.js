import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import Kernel from 'ml-kernel';
import { Matrix } from 'ml-matrix';
import { beforeAll, expect, test } from 'vitest';

import XtestData from '../../data/Xtest.json' with { type: 'json' };
import Xtest1Data from '../../data/Xtest1.json' with { type: 'json' };
import XtrainData from '../../data/Xtrain.json' with { type: 'json' };
import Xtrain1Data from '../../data/Xtrain1.json' with { type: 'json' };
import YtestData from '../../data/Ytest.json' with { type: 'json' };
import Ytest1Data from '../../data/Ytest1.json' with { type: 'json' };
import YtrainData from '../../data/Ytrain.json' with { type: 'json' };
import Ytrain1Data from '../../data/Ytrain1.json' with { type: 'json' };
import toData from '../../data/to.json' with { type: 'json' };
import tpData from '../../data/tp.json' with { type: 'json' };
import { KOPLS } from '../KOPLS.js';

expect.extend({ toBeDeepCloseTo });

let Xtest;
let Xtrain;
let Ytest;
let Ytrain;
let Tp;
let to;
let kernel;
let cls;

beforeAll(() => {
  Xtest = new Matrix(XtestData);
  Xtrain = new Matrix(XtrainData);
  Ytest = YtestData;
  Ytrain = new Matrix(YtrainData);
  Tp = new Matrix(tpData);
  to = new Matrix(toData);

  kernel = new Kernel('gaussian', {
    sigma: 25,
  });

  cls = new KOPLS({
    orthogonalComponents: 10,
    predictiveComponents: 1,
    kernel,
  });

  cls.train(Xtrain, Ytrain);
});

test('K-OPLS test with main features', () => {
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

test('Load and save', () => {
  // eslint-disable-next-line unicorn/prefer-structured-clone -- JSON round-trip exercises model serialization/deserialization
  let model = KOPLS.load(JSON.parse(JSON.stringify(cls)), kernel);
  let output = model.predict(Xtest).prediction;

  expect(output.to2DArray()).toBeDeepCloseTo(Ytest, 3);
});

test('with real dataset', () => {
  Xtest = new Matrix(Xtest1Data);
  Xtrain = new Matrix(Xtrain1Data);
  Ytest = Ytest1Data;
  Ytrain = new Matrix(Ytrain1Data);

  cls = new KOPLS({
    orthogonalComponents: 10,
    predictiveComponents: 2,
    kernel,
  });

  cls.train(Xtrain, Ytrain);
  let output = cls.predict(Xtest).prediction;

  expect(output.to2DArray()).toBeDeepCloseTo(Ytest, 1);
});
