import { Matrix } from 'ml-matrix';

import { plsNIPALS } from '../plsNIPALS';
import { tss } from '../utils';


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
    let y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    let model = plsNIPALS(x.center().scale('column'), y.center().scale());

    expect(model.scores.to1DArray()).toHaveLength(8);
    expect(model.loadings.to1DArray()).toHaveLength(4);
    expect(model.scores.sum()).toBeCloseTo(0, 10);
    expect(model.yRes.sum()).toBeCloseTo(0, 10);
    expect(model.xRes.sum('column')).toStrictEqual([0, 0, 0, 0]);
    expect(model.loadings.sum()).toBeCloseTo(-1.41421356237309, 10);
  });

  it('test pls-nipals simpleDataset with internal scaling', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let y = Matrix.from1DArray(8, 1, [1, 1, 1, 1, -2, -2, -2, -2]);
    let model = plsNIPALS(x, y, { scale: 'TRUE' });

    expect(model.scores.to1DArray()).toHaveLength(8);
    expect(model.loadings.to1DArray()).toHaveLength(4);
    expect(model.scores.sum()).toBeCloseTo(0, 10);
    expect(model.yRes.sum()).toBeCloseTo(0, 10);
    expect(model.xRes.sum('column')).toStrictEqual([0, 0, 0, 0]);
    expect(model.loadings.sum()).toBeCloseTo(-1.41421356237309, 10);
  });
});

