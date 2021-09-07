// get dataset-metadata
const {
  getNumbers,
  getClasses,
  getCrossValidationSets,
} = require('ml-dataset-iris');
const { METADATA } = require('ml-dataset-metadata');
const { Matrix } = require('ml-matrix');

// get frozen folds for testing purposes
let cvFolds = getCrossValidationSets(7, { idx: 0, by: 'trainTest' });

let x = new Matrix(iris);

let oplsOptions = { cvFolds };

// get labels as factor (for regression)
let labels = new METADATA([metadata], { headers: ['iris'] });
let y = labels.get('iris', { format: 'factor' }).values;

// get model
let model = new OPLS(x, y, oplsOptions);
