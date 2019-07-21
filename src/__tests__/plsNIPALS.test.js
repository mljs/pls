import { Matrix } from 'ml-matrix';

import { nipals } from '../nipals';

describe('pls-nipals', () => {
  it('test Matrix.scale()', () => {
    let y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    expect(y.scale().sum()).toBeCloseTo(-2.3664319132398464, 10);
  });
  it('test Matrix.center().scale()', () => {
    let y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    expect(y.center().scale().sum()).toBeCloseTo(0, 10);
  });
  it('test pls-nipals simpleDataset with external scaling', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let Y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    Y.center().scale();
    let model = nipals(x.center().scale('column'), { Y });

    expect(model.t.rows).toStrictEqual(8);
    expect(model.w.columns).toStrictEqual(4);
    expect(model.t.sum()).toBeCloseTo(0, 10);
    // expect(model.xResidual.get(0, 0)).toBeCloseTo(0, 10);
    expect(model.yResidual.sum()).toBeCloseTo(0, 10);
    expect(model.xResidual.sum('column')).toStrictEqual([0, 0, 0, 0]);
    expect(model.w.sum()).toBeCloseTo(-1.41421356237309, 10);
  });

  it('test pls-nipals simpleDataset with internal scaling', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let Y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    x = x.center('column').scale('column');
    Y = Y.center('column').scale('column');
    let model = nipals(x, { Y });

    expect(model.t.rows).toStrictEqual(8);
    expect(model.w.columns).toStrictEqual(4);
    expect(model.t.sum()).toBeCloseTo(0, 10);
    expect(model.yResidual.sum()).toBeCloseTo(0, 10);
    expect(model.xResidual.sum('column')).toStrictEqual([0, 0, 0, 0]);
    expect(model.w.sum()).toBeCloseTo(-1.41421356237309, 10);
  });
});

