# Partial Least Squares (PLS), Kernel-based Orthogonal Projections to Latent Structures (K-OPLS) and NIPALS based OPLS

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![npm download][download-image]][download-url]

PLS regression algorithm based on the Yi Cao implementation:

[PLS Matlab code](http://www.mathworks.com/matlabcentral/fileexchange/18760-partial-least-squares-and-discriminant-analysis)

K-OPLS regression algorithm based on [this paper](http://onlinelibrary.wiley.com/doi/10.1002/cem.1071/abstract).

[K-OPLS Matlab code](http://kopls.sourceforge.net/download.shtml)

OPLS implementation based on the R package [Metabomate](https://github.com/kimsche/MetaboMate) using NIPALS factorization loop.

## installation

`$ npm i ml-pls`

## Usage

### [PLS](./src/PLS.js)

```js
import PLS from 'ml-pls';

const X = [
  [0.1, 0.02],
  [0.25, 1.01],
  [0.95, 0.01],
  [1.01, 0.96],
];
const Y = [
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
];
const options = {
  latentVectors: 10,
  tolerance: 1e-4,
};

const pls = new PLS(options);
pls.train(X, Y);
```

### [OPLS-R](./src/OPLS.js)

```js
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
```

The OPLS class is intended for exploratory modeling, that is not for the creation of predictors. Therefore there is a built-in k-fold cross-validation loop and Q2y is an average over the folds.

```js
console.log(model.model[0].Q2y);
```
should give 0.9209227614652857

### [OPLS-DA](./src/OPLS.js)

```js
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
```

If for some reason a predictor is necessary the following code may serve as an example

### [Prediction](./src/OPLS.js)

```js
import {
  getNumbers,
  getClassesAsNumber,
  getCrossValidationSets,
} from 'ml-dataset-iris';
import { OPLS } from 'ml-pls';

// get frozen folds for testing purposes
const { testIndex, trainIndex } = getCrossValidationSets(7, {
  idx: 0,
  by: 'trainTest',
})[0];

// Getting the data of selected fold
const irisNumbers = getNumbers();
const testData = irisNumbers.filter((el, idx) => testIndex.includes(idx));
const trainingData = irisNumbers.filter((el, idx) => trainIndex.includes(idx));

// Getting the labels of selected fold
const irisLabels = getClassesAsNumber();
const testLabels = irisLabels.filter((el, idx) => testIndex.includes(idx));
const trainingLabels = irisLabels.filter((el, idx) => trainIndex.includes(idx));

const model = new OPLS(trainingData, trainingLabels);
console.log(model.mode); // 'discriminantAnalysis'
const prediction = model.predict(testData, { trueLabels: testLabels });
// Get the predicted Q2 value
console.log(prediction.Q2y); // 0.9247698398971457
```

### [K-OPLS](./src/KOPLS.js)

```js
import Kernel from 'ml-kernel';
import { KOPLS } from 'ml-pls';

const kernel = new Kernel('gaussian', {
  sigma: 25,
});

const X = [
  [0.1, 0.02],
  [0.25, 1.01],
  [0.95, 0.01],
  [1.01, 0.96],
];
const Y = [
  [1, 0],
  [1, 0],
  [1, 0],
  [0, 1],
];

const cls = new KOPLS({
  orthogonalComponents: 10,
  predictiveComponents: 1,
  kernel: kernel,
});

cls.train(X, Y);

const {
  prediction, // prediction
  predScoreMat, // Score matrix over prediction
  predYOrthVectors, // Y-Orthogonal vectors over prediction
} = cls.predict(X);

console.log(prediction);
console.log(predScoreMat);
console.log(predYOrthVectors);
```

## [API Documentation](http://mljs.github.io/pls/)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pls.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-pls
[ci-image]: https://github.com/mljs/pls/workflows/Node.js%20CI/badge.svg?branch=master
[ci-url]: https://github.com/mljs/pls/actions?query=workflow%3A%22Node.js+CI%22
[download-image]: https://img.shields.io/npm/dm/ml-pls.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-pls
