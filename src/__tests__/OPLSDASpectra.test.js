import { readFileSync } from 'fs';
import { join } from 'path';

import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { Matrix } from 'ml-matrix';
import { parse } from 'papaparse';

import kFoldCoffeeTest from '../../data/kFoldCoffeeTest.json';
import { OPLS } from '../OPLS.js';

expect.extend({ toBeDeepCloseTo });

const raw = readFileSync(join(__dirname, '../../data/coffee1k.tsv'), 'utf8');
const coffee1k = parse(raw, { delimiter: '\t', dynamicTyping: true }).data;
const y = [];
const x = [];
for (let j = 1; j < coffee1k.length - 1; j++) {
  y.push(coffee1k[j][1]);
  const spectrum = [];
  for (let i = 2; i < coffee1k[j].length; i++) {
    spectrum.push(coffee1k[j][i]);
  }
  x.push(spectrum);
}

/* R-code:
  library(MetaboMate)
  coffee1k <- read.table(file = '~/coffee1k.tsv', sep = '\t', header = TRUE)
  IDs <- coffee1k[,1]
  category <- coffee1k[,2]
  X <- coffee1k[,3:ncol(coffee1k)]
  model=opls(X, category)
*/

test('Statistic values of OPLS-DA working on a 2 classes spectra dataset', () => {
  const spectra = new Matrix(x);
  const opls = new OPLS(spectra, y, {
    center: true,
    scale: true,
    cvFolds: kFoldCoffeeTest,
  });

  /*
  model@summary
          R2X  R2Y   Q2 AUROC
  PC_o 1 0.35 0.97 0.94     1
  */
  expect(opls.output.auc).toStrictEqual([1, 1]);
  expect(opls.output.Q2y).toBeDeepCloseTo([0.94, 0.95], 2);
  expect(opls.output.R2y).toBeDeepCloseTo([0.97, 0.99], 2);
});
