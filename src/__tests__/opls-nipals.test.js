import { Matrix } from 'ml-matrix';

import { iris, oplsWrapper, OPLS } from '../opls-nipals';
import { shuffleArray, DataClass, Dataset } from '../utils';

describe('pls-nipals', () => {
  it('test pls-nipals iris data', () => {
    let rawData = iris();
    let metadata = rawData.map((d) => d[4]);
    expect(rawData).toHaveLength(150);
    expect(metadata).toHaveLength(150);
  });
  it('test pls-nipals permutation', () => {
    let rawData = iris();
    let metadata = rawData.map((d) => d[4]);
    let permutations = shuffleArray(metadata.slice(0));
    expect(permutations).toHaveLength(150);
  });
  it('test pls-nipals dataArray', () => {
    let rawData = iris();
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
    let classVector = irisDataset.getClass()[0].classMatrix;
    let resTot = OPLS(dataArray, classVector);
    // let resTot = oplsWrapper(irisDataset);
    // let scoresTot = resTot.scoresExport;
    console.log(resTot);
    expect(scoresTot.map((x) => x.scoresX)).toHaveLength(4);
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
