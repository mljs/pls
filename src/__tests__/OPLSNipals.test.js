import { Matrix } from 'ml-matrix';
import { getNumbers, getClasses } from 'ml-dataset-iris';
import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';

import { OPLSNipals } from '../OPLSNipals';

expect.extend({ toBeDeepCloseTo });

const iris = getNumbers();
const metadata = getClasses();

describe('opls-nipals', () => {
  it('test pls-nipals iris data', () => {
    expect(iris).toHaveLength(150);
    expect(metadata).toHaveLength(150);
  });
  it('test pls-nipals dataArray', () => {
    let y = Matrix.from1DArray(150, 1, metadata);
    let x = new Matrix(iris);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = OPLSNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(150);
  });
  it('test pls-nipals simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(rawData);
    let y = Matrix.from1DArray(8, 1, [1, 1, 2, 2, 3, 1, 3, 3]);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = OPLSNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(8);

    expect(model.weightsXPred.to1DArray()).toStrictEqual([0.5, -0.5, 0.5, 0.5]);
  });
});
