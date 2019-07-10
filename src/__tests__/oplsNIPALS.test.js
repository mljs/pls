import { Matrix } from 'ml-matrix';

import { oplsNIPALS } from '../oplsNIPALS';

describe('opls-nipals', () => {
  it('test pls-nipals iris data', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    expect(rawData).toHaveLength(150);
    expect(metadata).toHaveLength(150);
  });
  it('test pls-nipals dataArray', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    rawData = rawData.map((d) => d.slice(0, 4));
    let dataArray = new Matrix(150, 4);
    rawData.forEach((el, i) => dataArray.setRow(i, rawData[i]));
    let y = Matrix.from1DArray(150, 1, metadata);
    let x = dataArray;

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = oplsNIPALS(x, y);
    // let resTot = oplsWrapper(irisDataset);
    // let scoresTot = resTot.scoresExport;
    expect(model.scoresXOrtho.to1DArray()).toHaveLength(150);
  });
  it('test pls-nipals simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let y = Matrix.from1DArray(8, 1, [1, 1, 2, 2, 3, 1, 3, 3]);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = oplsNIPALS(x, y);
    expect(model.scoresXOrtho.to1DArray()).toHaveLength(8);
    expect(model.weightsPred.to1DArray()).toStrictEqual([0.5, -0.5, 0.5, 0.5]);
  });
});
