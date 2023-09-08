import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { ConfusionMatrix } from 'ml-confusion-matrix';
import { getNumbers, getClasses } from 'ml-dataset-iris';
import { Matrix } from 'ml-matrix';

import cvSets from '../../data/kFoldStratifiedTest.json';
import { OPLS } from '../OPLS';

// Cross-validation sets for R code in Metabomate
// cv_sets <- list(
//   c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,40,41,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,66,68,69,70,72,73,74,75,77,78,79,80,81,82,83,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,105,106,107,108,109,110,111,113,114,115,116,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,136,137,139,140,141,142,143,144,145,146,147,148,149,150),
//   c(3,4,5,6,7,8,9,10,11,12,13,15,16,18,19,21,22,23,24,25,26,27,28,29,30,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60,61,62,63,64,65,66,67,68,70,71,72,73,74,75,76,77,78,79,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,116,117,118,119,120,121,122,124,125,127,130,131,132,133,134,135,136,137,138,140,141,142,143,144,145,146,148,149,150),
//   c(1,2,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,36,37,38,39,40,41,42,44,45,46,47,50,51,52,53,55,56,57,58,59,60,61,62,64,65,66,67,68,69,70,71,72,73,74,75,76,78,79,80,81,82,83,84,85,86,88,89,90,91,92,93,95,96,97,98,99,100,101,103,104,106,108,109,111,112,114,115,117,119,120,121,123,124,125,126,127,128,129,130,131,132,134,135,136,138,139,140,141,142,143,145,146,147,149,150),
//   c(1,2,3,4,5,7,9,12,14,15,16,17,19,20,21,24,25,26,27,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,63,64,65,67,69,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,87,89,91,92,93,94,95,96,97,99,100,101,102,103,104,105,106,107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,125,126,127,128,129,130,131,133,134,135,136,137,138,139,140,141,142,143,144,145,147,148,149,150),
//   c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,28,29,31,32,33,34,35,36,37,39,40,42,43,44,45,46,48,49,51,53,54,58,59,61,62,63,65,66,67,68,69,70,71,72,74,75,76,77,79,80,81,83,84,85,86,87,88,89,90,94,96,97,98,100,101,102,103,104,105,107,108,109,110,112,113,115,116,117,118,120,122,123,124,125,126,127,128,129,130,131,132,133,135,137,138,139,140,141,143,144,146,147,148,149,150),
//   c(1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,76,77,78,80,81,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,121,122,123,124,125,126,127,128,129,132,133,134,135,136,137,138,139,142,144,145,146,147,148,149,150),
//   c(1,2,3,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,39,41,42,43,45,47,48,49,50,52,53,54,55,56,57,59,60,61,62,63,64,65,66,67,68,69,70,71,73,75,76,77,78,79,80,82,83,84,86,87,88,89,90,91,92,93,94,95,98,99,102,104,105,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,126,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148)
// )

expect.extend({ toBeDeepCloseTo });
const iris = getNumbers();
const metadata = getClasses();

/* # R-code:
library(MetaboMate)
data(iris)
X=as.matrix(iris[,1:4])
labels=cbind(as.character(iris[,5]))
model=opls(X, labels)
*/

const x = new Matrix(iris);
const opls = new OPLS(x, metadata, { cvFolds: cvSets });
describe('Statistic values with OPLS-DA working on iris', () => {
  // > model@summary
  //   R2X  R2Y   Q2 AUROC
  // PC_o 1 0.72 0.67 0.67  0.67
  it('Test statistic values', () => {
    // Multi-class area under the curve: 0.6704
    expect(opls.output.auc[0]).toBeCloseTo(0.6704, 4);
    // Multi-class area under the curve: 0.6732
    expect(opls.output.auc[1]).toBeCloseTo(0.6732, 4);
  });

  it('Test E matrix', () => {
    //   > model@E
    //       Sepal.Length Sepal.Width  Petal.Length  Petal.Width
    //  [1,]  0.056367716  0.19711813 -2.417647e-05  0.061137059
    //  [2,] -0.176379276 -0.77771969 -1.487345e-01 -0.128314741
    //  [3,] -0.055799091 -0.24308369 -4.437359e-02 -0.041734675
    //  [4,] -0.110771432 -0.36843166  1.732106e-02 -0.127486751
    //  [5,]  0.113426110  0.47315503  7.063183e-02  0.093097059
    const residualData = opls.output.residualData.to2DArray();
    expect(residualData[0]).toBeDeepCloseTo(
      [0.056367716, 0.19711813, -2.417647e-5, 0.061137059],
      8,
    );
    expect(residualData[1]).toBeDeepCloseTo(
      [-0.176379276, -0.77771969, -1.487345e-1, -0.128314741],
      7,
    );
    expect(residualData[2]).toBeDeepCloseTo(
      [-0.055799091, -0.24308369, -4.437359e-2, -0.041734675],
      8,
    );
    expect(residualData[3]).toBeDeepCloseTo(
      [-0.110771432, -0.36843166, 1.732106e-2, -0.127486751],
      8,
    );
    expect(residualData[4]).toBeDeepCloseTo(
      [0.11342611, 0.47315503, 7.063183e-2, 0.093097059],
      8,
    );

    // [146,]  0.2229086977  0.497928923 -2.596420e-01  0.351741065
    // [147,] -0.1304725091 -0.711272547 -2.354227e-01 -0.041794135
    // [148,]  0.1264911514  0.377266215 -5.990600e-02  0.162570429
    // [149,]  0.4457218985  1.694858777  1.261085e-01  0.430024961
    // [150,]  0.1037061687  0.549617725  1.726800e-01  0.039348872

    expect(residualData[145]).toBeDeepCloseTo(
      [0.2229086977, 0.497928923, -2.59642e-1, 0.351741065],
      7,
    );
    expect(residualData[146]).toBeDeepCloseTo(
      [-0.1304725091, -0.711272547, -2.354227e-1, -0.041794135],
      7,
    );
    expect(residualData[147]).toBeDeepCloseTo(
      [0.1264911514, 0.377266215, -5.9906e-2, 0.162570429],
      8,
    );
    expect(residualData[148]).toBeDeepCloseTo(
      [0.4457218985, 1.694858777, 1.261085e-1, 0.430024961],
      7,
    );
    expect(residualData[149]).toBeDeepCloseTo(
      [0.1037061687, 0.549617725, 1.7268e-1, 0.039348872],
      7,
    );
  });
});

describe('OPLS-DA test orthogonal values', () => {
  it('test orthogonal scores', () => {
    // > model@t_orth
    //             [,1]
    // [1,] -0.120705304
    // [2,]  0.001762035
    // [3,]  0.221496966
    // [4,]  0.323480471
    // [5,] -0.006342759
    expect(
      opls.output.orthogonalScores.to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [-0.120705304, 0.001762035, 0.221496966, 0.323480471, -0.006342759],
      8,
    );

    // [146,]  0.056756875
    // [147,]  0.074150505
    // [148,]  0.041028264
    // [149,]  0.631370238
    // [150,]  0.484405741
    expect(
      opls.output.orthogonalScores.to1DArray().slice(145, 150),
    ).toBeDeepCloseTo(
      [0.056756875, 0.074150505, 0.041028264, 0.631370238, 0.484405741],
      8,
    );
  });

  it('test orthogonal loadings', () => {
    // > model@p_orth
    //      Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]    -1.324534  -0.5627685   -0.3332748 0.004204019
    expect(opls.output.orthogonalLoadings.to1DArray()).toBeDeepCloseTo(
      [-1.324534, -0.5627685, -0.3332748, 0.004204019],
      7,
    );
  });

  it('test orthogonal weights', () => {
    // > model@w_orth
    //      Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]   -0.8212997   0.0661631    0.1587031   0.5439692
    expect(opls.output.orthogonalWeights.to1DArray()).toBeDeepCloseTo(
      [-0.8212997, 0.0661631, 0.1587031, 0.5439692],
      7,
    );
  });
});

describe('Test cross-validation scores', () => {
  it('Test predictive scores (predictiveScoresCV)', () => {
    // > model@t_cv
    //             [,1]
    // [1,]  2.38420136
    // [2,]  2.08072499
    // [3,]  2.25988084
    // [4,]  2.00857909
    // [5,]  2.38428588
    const predictiveScoresCV = opls.output.predictiveScoresCV;
    expect(
      predictiveScoresCV[predictiveScoresCV.length - 2].to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [2.38420136, 2.08072499, 2.25988084, 2.00857909, 2.38428588],
      8,
    );

    // [146,] -1.75510608
    // [147,] -1.62748669
    // [148,] -1.63785885
    // [149,] -1.72632549sP
    // [150,] -1.27648994
    expect(
      predictiveScoresCV[predictiveScoresCV.length - 2]
        .to1DArray()
        .slice(145, 150),
    ).toBeDeepCloseTo(
      [-1.75510608, -1.62748669, -1.63785885, -1.72632549, -1.27648994],
      8,
    );
  });

  it('Test predictive scores (orthogonalScoresCV)', () => {
    // > model@t_orth_cv
    // [1,] -0.1135301544
    // [2,] -0.0140748502
    // [3,]  0.1578422859
    // [4,]  0.3427612559
    // [5,]  0.0122617949
    const orthogonalScoresCV = opls.output.orthogonalScoresCV;
    expect(
      orthogonalScoresCV[orthogonalScoresCV.length - 2].to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [-0.1135301544, -0.0140748502, 0.1578422859, 0.3427612559, 0.0122617949],
      8,
    );

    // [146,]  0.0204401295
    // [147,]  0.0529381252
    // [148,]  0.1329515030
    // [149,]  0.6394495744
    // [150,]  0.4980731231
    expect(
      orthogonalScoresCV[orthogonalScoresCV.length - 2]
        .to1DArray()
        .slice(145, 150),
    ).toBeDeepCloseTo(
      [0.0204401295, 0.0529381252, 0.132951503, 0.6394495744, 0.4980731231],
      8,
    );
  });

  it('Test predictive scores (yHatScoresCV)', () => {
    // > model@t_yhat_cv
    //     [,1]
    // [1,]  5.234392e-03
    // [2,]  4.568125e-03
    // [3,] -1.154220e-02
    // [4,] -3.222685e-03
    // [5,] -3.825491e-03
    const yHatScoresCV = opls.output.yHatScoresCV;
    expect(
      yHatScoresCV[yHatScoresCV.length - 2].to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [5.234392e-3, 4.568125e-3, -1.15422e-2, -3.222685e-3, -3.825491e-3],
      8,
    );

    // [146,] -3.402418e-02
    // [147,] -3.573064e-03
    // [148,]  8.365263e-03
    // [149,]  2.769820e-03
    // [150,]  2.048077e-03
    expect(
      yHatScoresCV[yHatScoresCV.length - 2].to1DArray().slice(145, 150),
    ).toBeDeepCloseTo(
      [-3.402418e-2, -3.573064e-3, 8.365263e-3, 2.76982e-3, 2.048077e-3],
      8,
    );
  });
});

describe('OPLS-DA test predictive components', () => {
  it('Test predictive scores', () => {
    // > model@t_pred
    //             [,1]
    // [1,]  2.36197253
    // [2,]  2.03663185
    // [3,]  2.18730941
    // [4,]  2.04038657
    // [5,]  2.41783405
    expect(
      opls.output.predictiveComponents.to1DArray().slice(0, 5),
    ).toBeDeepCloseTo(
      [2.36197253, 2.03663185, 2.18730941, 2.04038657, 2.41783405],
      8,
    );

    // [146,] -1.88039825
    // [147,] -1.65429087
    // [148,] -1.52853204
    // [149,] -1.74143731
    // [150,] -1.28568876
    expect(
      opls.output.predictiveComponents.to1DArray().slice(145, 150),
    ).toBeDeepCloseTo(
      [-1.88039825, -1.65429087, -1.52853204, -1.74143731, -1.28568876],
      8,
    );
  });

  it('Test predictive loadings', () => {
    // > model@p_pred
    //      Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]   -0.4716058   0.3177661   -0.5825451  -0.5807357
    expect(opls.output.predictiveLoadings.to1DArray()).toBeDeepCloseTo(
      [-0.4716058, 0.3177661, -0.5825451, -0.5807357],
      7,
    );
  });

  it('Test predictive weights', () => {
    // > model@w_pred
    //      Sepal.Length Sepal.Width Petal.Length Petal.Width
    // [1,]   -0.4714635   0.3145122   -0.5860378  -0.5791061
    expect(opls.output.predictiveWeights.to1DArray()).toBeDeepCloseTo(
      [-0.4714635, 0.3145122, -0.5860378, -0.5791061],
      5,
    );
  });

  it('Test predictive betas', () => {
    // > model@betas_pred
    // [1] 0.7158934
    expect(opls.output.betas.to1DArray()).toBeDeepCloseTo([0.7158934], 7);
  });
});

describe('OPLS-DA test predict category', () => {
  const prediction = opls.predictCategory(x);
  it('Test setosa samples', () => {
    expect(prediction.slice(0, 5)).toStrictEqual(new Array(5).fill('setosa'));
  });

  it('Test versicolor samples', () => {
    expect(prediction.slice(50, 55)).toStrictEqual(
      new Array(5).fill('versicolor'),
    );
  });

  it('Test virginica samples', () => {
    expect(prediction.slice(100, 105)).toStrictEqual(
      new Array(5).fill('virginica'),
    );
  });

  it('Testing 1 sample', () => {
    const onePrediction = opls.predictCategory(x.getRow(0));
    expect(onePrediction).toStrictEqual(['setosa']);
  });

  it('Testing 2 samples', () => {
    const twoPrediction = opls.predictCategory([x.getRow(0), x.getRow(1)]);
    expect(twoPrediction).toStrictEqual(['setosa', 'setosa']);
  });

  it('Testing the accuracy with iris dataset', () => {
    const confusionMatrix = ConfusionMatrix.fromLabels(metadata, prediction);
    expect(confusionMatrix.getAccuracy()).toBeDeepCloseTo(0.973, 3);
  });
});
