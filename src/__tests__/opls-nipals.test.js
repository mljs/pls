import { Matrix } from 'ml-matrix';

import { iris, oplsWrapper, OPLS } from '../opls-nipals';
import { shuffleArray, DataClass, Dataset } from '../utils';

describe('opls-nipals', () => {
  it('test pls-nipals iris data', () => {
    let rawData = require('../../data/irisDataset.json');
    console.log(rawData);
    let metadata = rawData.map((d) => d[4]);
    expect(rawData).toHaveLength(150);
    expect(metadata).toHaveLength(150);
  });
  it('test pls-nipals permutation', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let permutations = shuffleArray(metadata.slice(0));
    expect(permutations).toHaveLength(150);
  });
  it('test pls-nipals dataArray', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let permutations = shuffleArray(metadata.slice(0));
    rawData = rawData.map((d) => d.slice(0, 4));
    let dataArray = new Matrix(150, 4);
    rawData.forEach((el, i) => dataArray.setRow(i, rawData[i]));
    let dataClass = DataClass('species', metadata).addClass('permutation', permutations).getClass();
    let irisDataset = Dataset({ dataMatrix: dataArray, options: {
      description: 'the famous iris dataset',
      dataClass: dataClass,
      metadata: metadata
    }
    });
    let y = irisDataset.getClass()[0].classMatrix;
    let x = irisDataset.summary().dataMatrix;
    let model = OPLS(x, y);
    // let resTot = oplsWrapper(irisDataset);
    // let scoresTot = resTot.scoresExport;
    expect(model.tOrtho.to1DArray()).toHaveLength(150);
  });
  it('test pls-nipals simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(8, 4);
    rawData.forEach((el, i) => x.setRow(i, rawData[i]));
    let y = Matrix.from1DArray(8, 1, [1, 1, 2, 2, 3, 1, 3, 3]);
    let model = OPLS(x, y);
    expect(model.tOrtho.to1DArray()).toHaveLength(8);
    expect(model.w.to1DArray()).toStrictEqual([0.5, -0.5, 0.5, 0.5]);
  });
});


/*

let scoresTot = resTot.getScores('circle', 2);
API.createData('totScoresPlot', scoresTot.scoresPlot);
API.createData('totPcaResult', scoresTot.scoresPlot[0].chart.chart);
API.createData('totPcaResult' + 'ellipse', scoresTot.scoresPlot[0].chart.ellipse);

let index = fh.sampleClass(classVector, 0.25);
console.log(index);

let newSet = data.sample(index);
console.log(newSet.train.summary(1));
console.log(newSet.test.summary(1));

let res = fh.oplsWrapper(newSet);
let scores = res.getScores('circle', 4);

API.createData('scoresPlot', scores.scoresPlot);
API.createData('pcaResult', scores.scoresPlot[0].chart.chart);
API.createData('pcaResult' + 'ellipse', scores.scoresPlot[0].chart.ellipse);

API.createData('testScoresPlot', scores.testScoresPlot);
API.createData('testPcaResult', scores.testScoresPlot[0].chart.chart);
 */
