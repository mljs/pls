import { Matrix } from 'ml-matrix';

import { OPLS } from '../OPLS.js';
import { sampleAClass, METADATA } from '../utils.js';

describe('OPLS', () => {
  it('test OPLS sampleAClass', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let c = sampleAClass(metadata, 0.1).trainIndex;
    let d = [];
    c.forEach((el) => d.push(metadata[el]));
    let counts = {};
    d.forEach((x) => {
      counts[x] = (counts[x] || 0) + 1;
    });
    expect(sampleAClass(metadata, 0.1).trainIndex).toHaveLength(15);
    expect(sampleAClass(metadata, 0.1).testIndex).toHaveLength(135);
    expect(sampleAClass(metadata, 0.1).mask).toHaveLength(150);
    expect(JSON.stringify(counts)).toBe(JSON.stringify({ setosa: 5, versicolor: 5, virginica: 5 }));
  });
  it('test OPLS sampleAClass 2', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let c = sampleAClass(metadata, 0.1).mask;

    let train = metadata.filter((f, idx) => c[idx]);
    let test = metadata.filter((f, idx) => !c[idx]);
    expect(train).toHaveLength(15);
    expect(test).toHaveLength(135);
    expect(sampleAClass(metadata, 0.1).mask).toHaveLength(150);
  });
  it('test LABELS', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let L = new METADATA().addMetadata('iris', metadata);
    expect(L.getMetadata('iris').nClass).toStrictEqual(3);
    expect(L.getMetadata('iris').classMatrix.rows).toStrictEqual(150);
  });
  it('test OPLS dataArray', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    rawData = rawData.map((d) => d.slice(0, 4));
    let x = new Matrix(150, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));

    let y = Matrix.from1DArray(150, 1, metadata);
    let trainTestLabels = require('../../data/trainTestLabels.json');
    // let trainTestLabels = sampleAClass(metadata, 0.8);
    // console.log(JSON.stringify(trainTestLabels));
    let options = { trainTestLabels, nComp: 6 };
    let model = OPLS(x, metadata, options);
    console.log(model);
    expect(model.R2Y).toHaveLength(5);
    expect(model.scoresExport).toHaveLength(5);
    expect(model.scoresExport[0].scoresX).toHaveLength(120);
    expect(model.Q2[0]).toBeCloseTo(0.941299692671602, 10);
    expect(model.Q2[4]).toBeCloseTo(0.43212543306026796, 10);
  });
  it('test OPLS simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let y = [1, 1, 2, 2, 3, 1, 3, 3];
    let model = OPLS(x, y, { nComp: 3, trainFraction: 0 });
    console.log(model);
    expect(model.R2Y).toHaveLength(2);
  });
});

