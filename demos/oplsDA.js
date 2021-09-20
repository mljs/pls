/* eslint-disable import/no-unresolved */
/* eslint-disable no-console */

import {
  getNumbers,
  getClasses,
  getCrossValidationSets,
} from 'ml-dataset-iris';
import { OPLS } from 'ml-pls';

const cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });
const data = getNumbers();
const irisLabels = getClasses();

const model = new OPLS(data, irisLabels, { cvFolds });
console.log(model.mode); // 'discriminantAnalysis'
console.log(model.model[0].auc); // 0.5366666666666665,
