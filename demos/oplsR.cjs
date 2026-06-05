/* eslint-disable import/no-unresolved */
/* eslint-disable no-console */

import {
  getNumbers,
  getClassesAsNumber,
  getCrossValidationSets,
} from 'ml-dataset-iris';
import { OPLS } from 'ml-pls';

const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
const data = getNumbers();
const irisLabels = getClassesAsNumber();

const model = new OPLS(data, irisLabels, { cvFolds });
console.log(model.mode); // 'regression'
console.log(model.model[0].Q2Y); // 0.9209227614652857,