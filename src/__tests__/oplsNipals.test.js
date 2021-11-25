import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { getNumbers, getClasses } from 'ml-dataset-iris';
import { METADATA } from 'ml-dataset-metadata';
import { Matrix } from 'ml-matrix';

import { oplsNipals } from '../oplsNipals';

expect.extend({ toBeDeepCloseTo });

const iris = getNumbers();
const metadata = getClasses();
const newM = new METADATA([metadata], { headers: ['iris'] });
let numericValues = newM.get('iris', { format: 'matrix' }).values;

// R - Code
// library(MetaboMate)
// data(iris)
// X=as.matrix(iris[,1:4])
// labels=cbind(as.character(iris[,5]))

// create_dummy_Y=function(Y){
//   if(!is.numeric(Y)){
//     Y_levels=unique(Y)
//     if(length(Y_levels)==2){Y_new=cbind(as.numeric(as.factor(Y)))}else{
//       Y_new=matrix(-1, nrow=length(Y), ncol=length(Y_levels))
//       for(i in 1:length(Y_levels)){
//         Y_new[which(Y==Y_levels[i]),i]=1
//       }
//       colnames(Y_new)=Y_levels}
//     Y_levs=unique(data.frame(Original=Y, Numeric=Y_new, stringsAsFactors = F))
//     # return(list(Y_new, Y_levs))
//     return(list(Y_new, Y_levs))}else{return(list(cbind(Y), data.frame()))}
// }

// Y1 <- create_dummy_Y(labels)
// Y <- Y1[[1]]
// model=NIPALS_OPLS_component_mulitlevel(X, Y)

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

    let model = oplsNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(150);
  });

  it('test opls-nipals with iris', () => {
    let x = new Matrix(iris);
    let y = numericValues.clone();
    let model = oplsNipals(x, y);

    expect(model.scoresXpred.getRow(0)).toBeDeepCloseTo([5.66576130528549]);
    expect(model.scoresXpred.getRow(1)).toBeDeepCloseTo([5.357904200567914]);
    expect(model.scoresXpred.getRow(148)).toBeDeepCloseTo([9.161531522704093]);
    expect(model.scoresXpred.getRow(149)).toBeDeepCloseTo([8.539939678663997]);

    expect(model.loadingsXpred.getRow(0)).toBeDeepCloseTo([
      0.7533846125482135, 0.37991165000540095, 0.5175455876998517,
      0.1697637836019074,
    ]);

    expect(model.weightsXPred.getColumn(0)).toBeDeepCloseTo([
      0.7191635887943787, 0.3280487739174021, 0.5781304613979087,
      0.20236823883090044,
    ]);

    expect(model.scoresXOrtho.getRow(0)).toBeDeepCloseTo([2.8553426763334433]);
    expect(model.scoresXOrtho.getRow(1)).toBeDeepCloseTo([2.501798988578702]);
    expect(model.scoresXOrtho.getRow(148)).toBeDeepCloseTo([-0.14717957843937]);
    expect(model.scoresXOrtho.getRow(149)).toBeDeepCloseTo([-0.1097900904882]);

    expect(model.loadingsXOrtho.getRow(0)).toBeDeepCloseTo([
      1.893676431872939, 1.2763813703908802, 0.5781825736661366,
      0.10014507134716187,
    ]);

    expect(model.weightsXOrtho.getRow(0)).toBeDeepCloseTo([
      0.36913469592136616, 0.5594334971409369, -0.6535157770509538,
      -0.35169712493733896,
    ]);

    expect(model.loadingsY.getRow(0)).toBeDeepCloseTo([0.14558532608468372]);
  });

  it('test opls-nipals with iris multi Y', () => {
    let x = new Matrix(iris);
    let y = new Matrix(createDummyY(metadata)).transpose();
    let model = oplsNipals(x, y);
    // > model$`Loadings X pred`
    //     Sepal.Length   Sepal.Width  Petal.Length  Petal.Width
    // [1,] 4.552819e-05 -4.845046e-05 -5.427571e-05 7.195014e-05
    expect(model.loadingsXpred.getRow(0)).toBeDeepCloseTo(
      [4.552819e-5, -4.845046e-5, -5.427571e-5, 7.195014e-5],
      6,
    );
    // > model$`Weights pred`
    //     [,1]
    // Sepal.Length -0.7191594
    // Sepal.Width  -0.3262782
    // Petal.Length -0.5792557
    // Petal.Width  -0.2020275
    expect(model.weightsXPred.getColumn(0)).toBeDeepCloseTo(
      [-0.7191594, -0.3262782, -0.5792557, -0.2020275],
      6,
    );
    // > model$`Scores X pred`
    //     [,1]
    // [1,]  -5.661050
    // [2,]  -5.354079
    // [3,]  -5.217577
    // [4,]  -5.228885
    // [5,]  -5.621762
    expect(model.scoresXpred.getColumn(0).slice().splice(0, 5)).toBeDeepCloseTo(
      [-5.66105, -5.354079, -5.217577, -5.228885, -5.621762],
      6,
    );
    // [146,]  -9.273996
    // [147,]  -8.626531
    // [148,]  -9.069555
    // [149,]  -9.160778
    // [150,]  -8.539729
    expect(
      model.scoresXpred.getColumn(0).slice().splice(145, 149),
    ).toBeDeepCloseTo(
      [-9.273996, -8.626531, -9.069555, -9.160778, -8.539729],
      6,
    );
    // > model$`Loadings X orth`
    //     Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]    0.9095791  -0.4465771   -0.3667357   0.4036415
    expect(model.loadingsXOrtho.getRow(0)).toBeDeepCloseTo(
      [0.9095791, -0.4465771, -0.3667357, 0.4036415],
      7,
    );
    // > model$`Weights X orth`
    //     Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]    0.4065189  -0.4326117   -0.4846251   0.6424391
    expect(model.weightsXOrtho.getRow(0)).toBeDeepCloseTo(
      [0.4065189, -0.4326117, -0.4846251, 0.6424391],
      7,
    );
    // > model$`Scores X orth`
    //     [,1]
    // [1,]  0.0091181654
    // [2,]  0.1441202375
    // [3,]  0.0247566222
    // [4,] -0.0695591173
    // [5,] -0.0747948970
    expect(
      model.scoresXOrtho.getColumn(0).slice().splice(0, 5),
    ).toBeDeepCloseTo(
      [0.0091181654, 0.1441202375, 0.0247566222, -0.0695591173, -0.074794897],
      9,
    );

    // [146,]  0.3834010337
    // [147,]  0.2770487017
    // [148,]  0.1093655207
    // [149,] -0.0898281264
    // [150,] -0.2145711385
    expect(
      model.scoresXOrtho.getColumn(0).slice().splice(145, 149),
    ).toBeDeepCloseTo(
      [0.3834010337, 0.2770487017, 0.1093655207, -0.0898281264, -0.2145711385],
      9,
    );
    // > model$`Loadings Y`
    //     setosa versicolor  virginica
    // [1,] 0.06371221 0.03858069 0.02296009
    expect(model.loadingsY.getRow(0)).toBeDeepCloseTo(
      [0.06371221, 0.03858069, 0.02296009],
      8,
    );
  });

  it('test pls-nipals simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(rawData);
    let y = Matrix.from1DArray(8, 1, [1, 1, 2, 2, 3, 1, 3, 3]);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = oplsNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(8);

    expect(model.weightsXPred.to1DArray()).toStrictEqual([0.5, -0.5, 0.5, 0.5]);
  });
});

function createDummyY(array) {
  const features = [...new Set(array)];
  const result = [];
  if (features.length > 2) {
    for (let i = 0; i < features.length; i++) {
      const feature = [];
      for (let j = 0; j < array.length; j++) {
        const point = features[i] === array[j] ? 1 : -1;
        feature.push(point);
      }
      result.push(feature);
    }
    return result;
  } else {
    const result = [];
    for (let j = 0; j < array.length; j++) {
      const point = features[0] === array[j] ? 2 : 1;
      result.push(point);
    }
    return [result];
  }
}
