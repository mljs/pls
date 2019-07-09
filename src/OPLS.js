import Matrix from 'ml-matrix';

import { plsNIPALS } from './plsNIPALS.js';
import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, sampleAClass, summaryMetadata } from './utils.js';

export const OPLS = (features, labels, options = {}) => {
  const {
    trainFraction = 0.7,
    trainTestLabels,
    nComp = 6,
    center = true,
    scale = true,
    observations = Array(features.rows).fill(null).map((x, i) => `OBS${i + 1}`),
    variables = Array(features.columns).fill(null).map((x, i) => `VAR${i + 1}`),
  } = options;

  let matrixYData = features;
  let dataClass = summaryMetadata(labels).classMatrix;

  if (center) {
    matrixYData = matrixYData.center('column');
    dataClass = dataClass.center();
  }

  if (scale) {
    matrixYData.scale('column');
    dataClass = dataClass.scale();
  }

  let { testMatrixYData,
    testDataClass,
    trainMatrixYData,
    trainDataClass,
    index } = [];

  if (trainFraction !== 0) {
    if (!trainTestLabels) {
      index = sampleAClass(labels, trainFraction);
    } else {
      index = trainTestLabels;
    }
    // create test set
    testMatrixYData = new Matrix(index.testIndex.length, variables.length);
    testDataClass = new Matrix(index.testIndex.length, 1);
    index.testIndex.forEach((el, idx) => {
      testMatrixYData.setRow(idx, matrixYData.getRow(el));
      testDataClass.setRow(idx, dataClass.getRow(el));
    });
    // create train set
    trainMatrixYData = new Matrix(index.trainIndex.length, variables.length);
    trainDataClass = new Matrix(index.trainIndex.length, 1);
    index.trainIndex.forEach((el, idx) => {
      trainMatrixYData.setRow(idx, matrixYData.getRow(el));
      trainDataClass.setRow(idx, dataClass.getRow(el));
    });
  } else {
    trainMatrixYData = matrixYData;
    trainDataClass = dataClass;
  }

  let { xResidual,
    testxResidual,
    scores,
    testScores,
    Eh,
    testEh,
    Yhat,
    testYhat,
    oplsResult,
    plsComp,
    testPlsComp,
    testOpls,
    scoresExport = [],
    testScoresExport = [],
    R2X = [],
    R2Y = [],
    Q2 = [],
    oplsResultV = [],
    plsCompV = [],
    testPlsCompV = [] } = [];

  for (let i = 1; i < nComp; i++) {
    if (i === 1) {
      oplsResult = oplsNIPALS(trainMatrixYData, trainDataClass);
      if (trainFraction !== 0) {
        testOpls = oplsNIPALS(testMatrixYData, testDataClass);
      }
    } else {
      oplsResult = oplsNIPALS(xResidual, trainDataClass);
      if (trainFraction !== 0) {
        testOpls = oplsNIPALS(testxResidual, testDataClass);
      }
    }

    oplsResultV.push(oplsResult);
    xResidual = oplsResult.err;

    if (trainFraction !== 0) {
      testxResidual = testOpls.err;
    }

    plsComp = plsNIPALS(xResidual, trainDataClass);
    plsCompV.push(plsComp);

    if (trainFraction !== 0) {
      testPlsComp = plsNIPALS(testxResidual.clone(), testDataClass.clone());
      testPlsCompV.push(testPlsComp);
    }

    scores = xResidual.mmul(plsComp.weights.transpose()); // ok
    Eh = xResidual.clone().sub(scores.clone().mmul(plsComp.loadings)); // ok
    Yhat = scores.clone().mul(plsComp.betas); // ok

    if (trainFraction !== 0) {
      testScores = testxResidual.mmul(testPlsComp.weights.transpose()); // ok
      testEh = testxResidual.clone().sub(testScores.clone().mmul(testPlsComp.loadings)); // ok
      testYhat = testScores.clone().mul(testPlsComp.betas); // ok
    }

    if (trainFraction !== 0) {
      let testTssy = tss(testDataClass.clone());

      let testRss = testDataClass.clone().sub(testYhat);

      testRss = testRss.clone().mul(testRss).sum();

      let Q2y = 1 - (testRss / testTssy);
      Q2.push(Q2y);
    }

    let tssy = tss(trainDataClass.clone());
    let rss = trainDataClass.clone().sub(Yhat);
    rss = rss.clone().mul(rss).sum();
    let R2y = 1 - (rss / tssy);
    R2Y.push(R2y);

    let tssx = tss(trainMatrixYData.clone());
    let xEx = plsComp.scores.clone().mmul(plsComp.loadings);
    let rssx = tss(xEx.clone());
    let R2x = rssx / tssx;
    R2X.push(R2x);

    // store scores for export
    scoresExport.push({
      scoresX: scores.to1DArray(),
      scoresY: oplsResult.tOrtho.to1DArray()
    });

    if (trainFraction !== 0) {
      testScoresExport.push({
        scoresX: testScores.to1DArray(),
        scoresY: testOpls.tOrtho.to1DArray()
      });
    }

    // console.log(`OPLS iteration over #ofComponents: ${i}`);
  } // end of loop

  let eigenVectorMatrix = new Matrix(plsCompV.length, variables.length);
  plsCompV.forEach((e, i) => eigenVectorMatrix.setRow(i, e.loadings));

  let eigenVectors = new Array(eigenVectorMatrix.rows);

  let xAxis;
  if (typeof (variables[0]) === 'number') {
    xAxis = variables;
  } else {
    xAxis = variables.map((x, i) => i + 1);
  }

  for (let i = 0, l = eigenVectors.length; i < l; i++) {
    eigenVectors[i] = { id: [i + 1],
      eigenVal: null,
      data: { x: xAxis, y: eigenVectorMatrix[i] } };
  }

  return ({
    R2X,
    R2Y,
    scoresExport,
    eigenVectors,
    Q2,
    observations

    /* getScores(markers, radius) {
        let scoresPlot = [];
        scoresExport.forEach((e, i) => {
          let splot = plot(e.scoresX, e.scoresY, observations, train.getClass()[0], markers, radius);
          scoresPlot.push({ index: i + 1, chart: splot });
        });

        // API.createData('pcaResult', scoresPlot[0].chart.chart);
        // API.createData('pcaResult' + 'ellipse', scoresPlot[0].chart.ellipse);
        // API.createData('scoresPlot', scoresPlot);

        let testScoresPlot = [];
        if (test) {
          testScoresExport.forEach((e, i) => {
            let splot = plot(e.scoresX, e.scoresY, observations, test.getClass()[0], markers, radius * 2);
            testScoresPlot.push({ index: i + 1, chart: splot });
          });

          // API.createData('testPcaResult', testScoresPlot[0].chart.chart);
          // API.createData('testScoresPlot', testScoresPlot);
        }
        if (test) {
          return { scoresPlot, testScoresPlot };
          // return {scoresExport, testScoresExport};
        } else {
          return { scoresPlot };
          // return {scoresExport};
        }

        //   console.log('scores');
        //   let scorePlot = plot(scoresExport[nc - 1].scoresX,
        //     scoresExport[nc - 1].scoresY,
        //     observations,
        //     train.getClass()[0]
        //   );

        //   API.createData('pcaResult', scorePlot.chart);
        //   API.createData('pcaResult' + 'ellipse', scorePlot.ellipse);

        //   return scoresExport;
      } */
  });
};

