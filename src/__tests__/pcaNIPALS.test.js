import { Matrix, correlation } from 'ml-matrix';
import { Stat, array } from 'ml-stat';


import { pcaNIPALS } from '../pcaNIPALS';

describe('opls-nipals', () => {
  it('test pls-nipals dataArray', () => {
    let rawData = require('../../data/irisDataset.json');
    rawData = rawData.map((d) => d.slice(0, 4));
    let dataArray = new Matrix(150, 4);
    rawData.forEach((el, i) => dataArray.setRow(i, rawData[i]));
    let x = dataArray;

    x = x.center('column').scale('column');

    let model = pcaNIPALS(x);
    let model2 = pcaNIPALS(model.residual);
    let model3 = pcaNIPALS(model2.residual);
    let model4 = pcaNIPALS(model3.residual);

    // console.log('scores', JSON.stringify(model.scores.to1DArray()));
    // console.log('scores', JSON.stringify(model2.scores.to1DArray()));

    let irisPC = require('../../data/irisPC1-4.json');
    /* data("iris");
    metadata = iris[,5]
    dataMatrix = iris[,1:4]
    X = dataMatrix
    Xcs = scale(as.matrix(X),center=TRUE)
    pca = prcomp(Xcs)
    plot(pca$x[,1], pca$x[,2], col=c(rep(1,50), rep(2,50), rep(3,50)))

    library(jsonlite)
    toJSON(t(pca$x)) */
    let corr = correlation(model.scores, Matrix.from1DArray(150, 1, irisPC[0]));
    let corr2 = correlation(model2.scores, Matrix.from1DArray(150, 1, irisPC[1]));
    let corr3 = correlation(model3.scores, Matrix.from1DArray(150, 1, irisPC[2]));
    let corr4 = correlation(model4.scores, Matrix.from1DArray(150, 1, irisPC[3]));
    expect(model.scores.to1DArray()).toHaveLength(150);
    expect(corr.get(0, 0)).toBeCloseTo(-1, 6);
    expect(corr2.get(0, 0)).toBeCloseTo(-1, 6);
    expect(corr3.get(0, 0)).toBeCloseTo(-1, 2);
    expect(corr4.get(0, 0)).toBeCloseTo(-1, 2);
  });
});
