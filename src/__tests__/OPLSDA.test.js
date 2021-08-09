import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { getNumbers, getClasses } from 'ml-dataset-iris';
import { METADATA } from 'ml-dataset-metadata';
import { Matrix } from 'ml-matrix';

import kFoldStratified from '../../data/kFoldStratifiedTest.json';
import { OPLS } from '../OPLS.js';

expect.extend({ toBeDeepCloseTo });
const iris = getNumbers();
const metadata = getClasses();
const newM = new METADATA([metadata], { headers: ['iris'] });

/* # R-code:
library(MetaboMate)
data(iris)
X=as.matrix(iris[,1:4])
labels=cbind(as.character(iris[,5]))
model=opls(X, labels)
*/

/*
model@summary
       R2X  R2Y   Q2
PC_o 1 0.7 0.93 0.93
*/

test('Statistic values with OPLS-R working on iris dataset', () => {
  const x = new Matrix(iris);
  const y = newM.get('iris', { format: 'factor' }).values;
  const opls = new OPLS(x, y, { cvFolds: kFoldStratified });
  expect(opls.output.Q2y[0]).toBeCloseTo(0.93, 1);
  expect(opls.output.Q2y[1]).toBeCloseTo(0.93, 1);
  expect(opls.output.R2y[0]).toBeCloseTo(0.93, 1);
  expect(opls.output.R2y[1]).toBeCloseTo(0.93, 1);
});

/*
model@summary
       R2X  R2Y   Q2 AUROC
PC_o 1 0.72 0.67 0.67  0.57
*/
test('Statistic values with OPLS-DA working on iris dataset', () => {
  const x = new Matrix(iris);
  const opls = new OPLS(x, metadata, { cvFolds: kFoldStratified });
  expect(opls.output.Q2y[0]).toBeCloseTo(0.6666553, 6);
  expect(opls.output.Q2y[1]).toBeCloseTo(0.6666626, 6);
  expect(opls.output.auc[0]).toBeCloseTo(0.57, 1);
  expect(opls.output.auc[1]).toBeCloseTo(0.5, 1);
  expect(opls.output.R2y[0]).toBeCloseTo(0.67, 1);
  expect(opls.output.R2y[1]).toBeCloseTo(0.67, 1);
});
