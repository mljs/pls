import Matrix from 'ml-matrix';
// import CV from 'ml-cross-validation';
import { array } from 'ml-stat';

import { plsNIPALS } from './plsNIPALS.js';
import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, sampleAClass, summaryMetadata, getFolds } from './utils.js';

export class OPLS {
  constructor(features, labels, options = {}) {
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

    // check and remove for features with sd = 0 TODO here
    // check opls.R line 70

    let { testMatrixYData,
      testDataClass,
      trainMatrixYData,
      trainDataClass,
      testDataLabels,
      trainDataLabels,
      index } = [];

    if (trainFraction !== 0) {
      if (!trainTestLabels) {
        index = sampleAClass(labels, trainFraction);
      } else {
        index = trainTestLabels;
      }
      // create test set
      testMatrixYData = getTrainTest(matrixYData, dataClass, index).testFeatures;
      testDataClass = getTrainTest(matrixYData, dataClass, index).testLabels;
      testDataLabels = labels.filter((el, idx) => index.testIndex.includes(idx));
      // create train set
      trainMatrixYData = getTrainTest(matrixYData, dataClass, index).trainFeatures;
      trainDataClass = getTrainTest(matrixYData, dataClass, index).trainLabels;
      trainDataLabels = labels.filter((el, idx) => index.trainIndex.includes(idx));
    } else {
      trainMatrixYData = matrixYData;
      trainDataClass = dataClass;
    }

    this.R2Y = [];
    this.R2X = [];
    this.Q2 = [];
    let Q2k = [];
    let RR;
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
      oplsResultV = [],
      plsCompV = [],
      testPlsCompV = [] } = [];

    for (let i = 1; i < nComp; i++) {
      let folds = getFolds(trainDataLabels, 5);
      let { testMatrixYDataK,
        trainMatrixYDataK,
        testDataClassK,
        trainDataClassK } = [];

      Q2k = [];
      for (let k of folds) {
        testMatrixYDataK = getTrainTest(trainMatrixYData, dataClass, k).testFeatures;
        testDataClassK = getTrainTest(trainMatrixYData, dataClass, k).testLabels;

        trainMatrixYDataK = getTrainTest(trainMatrixYData, dataClass, k).trainFeatures;
        trainDataClassK = getTrainTest(trainMatrixYData, dataClass, k).trainLabels;

        if (i === 1) {
          oplsResult = oplsNIPALS(trainMatrixYDataK, trainDataClassK);
        } else {
          oplsResult = oplsNIPALS(xResidual, trainDataClassK);
        }

        oplsResultV.push(oplsResult);
        xResidual = oplsResult.filteredX;

        plsComp = plsNIPALS(xResidual, trainDataClassK);
        plsCompV.push(plsComp);

        Eh = testMatrixYDataK;
        // removing the orthogonal components from PLS
        for (let idx = 0; idx < i; idx++) {
          scores = Eh.mmul(plsComp.weights.transpose()); // ok
          Eh = Eh.clone().sub(scores.clone().mmul(plsComp.loadings)); // ok
        }
        // prediction
        Yhat = scores.clone().mul(plsComp.betas); // ok

        // calculate Q2y for each folds
        // ROC for DA is not implmented (check opls.R line 183) TODO
        let testTssy = tss(testDataClassK.clone());
        let testRss = testDataClassK.clone().sub(Yhat);
        testRss = testRss.clone().mul(testRss).sum();
        let Q2y = 1 - (testRss / testTssy);
        Q2k.push({ folds: k, Q2y });
      }
      this.Q2.push({ i, Q2: Q2k, mean: array.mean(Q2k.map((x) => x.Q2y)) });
      console.log(this.Q2.map((x) => x.mean));
      // console.log(this.R2X.map((x) => x.R2x).reduce((a, b) => a + b).toFixed(3));

      // calculate the R2y for the complete data

      if (i === 1) {
        RR = this.predictAll(trainMatrixYData, trainDataClass);
      } else {
        RR = this.predictAll(RR.xRes, trainDataClass);
      }
      this.R2Y.push(RR.R2y);
      console.log(this.R2Y);


      /*         let tssx = tss(trainMatrixYDataK.clone());
        let xEx = plsComp.scores.clone().mmul(plsComp.loadings);
        let rssx = tss(xEx.clone());
        let R2x = rssx / tssx;
        this.R2X.push({ k, R2x }); */

      if (trainFraction !== 0) {
        if (i === 1) {
          testOpls = oplsNIPALS(testMatrixYData, testDataClass);
        } else {
          testOpls = oplsNIPALS(testxResidual, testDataClass);
        }

        testxResidual = testOpls.filteredX;
        testPlsComp = plsNIPALS(testxResidual.clone(), testDataClass.clone());
        testPlsCompV.push(testPlsComp);
        testEh = testMatrixYData;
        for (let idx = 0; idx < i; idx++) {
          testScores = testEh.mmul(testPlsComp.weights.transpose()); // ok
          testEh = testEh.clone().sub(testScores.clone().mmul(testPlsComp.loadings)); // ok
        }
        testYhat = testScores.clone().mul(testPlsComp.betas); // ok
        let testTssyt = tss(testDataClass.clone());
        let testRsst = testDataClass.clone().sub(testYhat);
        testRsst = testRsst.clone().mul(testRsst).sum();
        let Q2yt = 1 - (testRsst / testTssyt);
        console.log(Q2yt);
      }
      // store scores for export
      scoresExport.push({
        scoresX: scores.to1DArray(),
        scoresY: oplsResult.scoresXOrtho.to1DArray()
      });

      if (trainFraction !== 0) {
        testScoresExport.push({
          scoresX: testScores.to1DArray(),
          scoresY: testOpls.scoresXOrtho.to1DArray()
        });
      }

      console.log(`OPLS iteration over #ofComponents: ${i}`);
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
  }
  predictAll(features, labels) {
    let res = oplsNIPALS(features, labels);
    let xRes = res.filteredX;
    let plsC = plsNIPALS(xRes, labels);
    // removing the orthogonal components from PLS
    let scores = features.mmul(plsC.weights.transpose()); // ok
    let Eh = features.clone().sub(scores.clone().mmul(plsC.loadings)); // ok
    // prediction
    let Yhat = scores.clone().mul(plsC.betas); // ok
    let tssy = tss(labels.clone());
    let rss = labels.clone().sub(Yhat);
    rss = rss.clone().mul(rss).sum();
    let R2y = 1 - (rss / tssy);
    return { R2y, xRes };
  }
  getResult() {
    console.log(this.R2Y);
    return this.R2Y;
    /*
    R2Y,
    Q2,
    scoresExport,
    eigenVectors,
    observations; */

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

      }

            CV.kFold(matrixYData, dataClass, function (trainFeatures, trainLabels, testFeatures) {
        let trainOPLS = oplsNIPALS(trainFeatures, trainLabels);
        let plsComponent = plsNIPALS(trainOPLS.filteredX, trainLabels);
        let scores = testFeatures.mmul(plsComponent.weights.transpose()); // ok
        console.log(testFeatures);
        // let Eh = testFeatures.clone().sub(scores.clone().mmul(plsComponent.loadings)); // ok
        let Yhat = scores.clone().mul(plsComponent.betas); // ok
        let testTssy = tss(testDataClass.clone());
        let testRss = testDataClass.clone().sub(Yhat);
        testRss = testRss.clone().mul(testRss).sum();
        let Q2y = 1 - (testRss / testTssy);
        console.log(Yhat);
        return Yhat;
        });*/
  }
}

function getTrainTest(matrixYData, dataClass, index) {
  let testFeatures = new Matrix(index.testIndex.length, matrixYData.columns);
  let testLabels = new Matrix(index.testIndex.length, 1);
  index.testIndex.forEach((el, idx) => {
    testFeatures.setRow(idx, matrixYData.getRow(el));
    testLabels.setRow(idx, dataClass.getRow(el));
  });

  let trainFeatures = new Matrix(index.trainIndex.length, matrixYData.columns);
  let trainLabels = new Matrix(index.trainIndex.length, 1);
  index.trainIndex.forEach((el, idx) => {
    trainFeatures.setRow(idx, matrixYData.getRow(el));
    trainLabels.setRow(idx, dataClass.getRow(el));
  });

  return ({
    trainFeatures,
    testFeatures,
    trainLabels,
    testLabels
  });
}
