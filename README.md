# Partial Least Squares (PLS) and Kernel-based Orthogonal Projections to Latent Structures (K-OPLS)

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

PLS regression algorithm based on the Yi Cao implementation:

[PLS Matlab code](http://www.mathworks.com/matlabcentral/fileexchange/18760-partial-least-squares-and-discriminant-analysis)

K-OPLS regression algorithm based on [this paper](http://onlinelibrary.wiley.com/doi/10.1002/cem.1071/abstract).

[K-OPLS Matlab code](http://kopls.sourceforge.net/download.shtml)

## installation

`$ npm install ml-pls`

## Usage

### [PLS](./src/pls.js)

```js
import PLS from 'ml-pls'

var X = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
var Y = [[1, 0], [1, 0], [1, 0], [0, 1]];
var options = {
  latentVectors: 10,
  tolerance: 1e-4
};

var pls = new PLS(options);
pls.train(X, Y);
```

### [K-OPLS](./src/kopls.js)

```js
// assuming that you created Xtrain, Xtest, Ytrain, Ytest

import Kernel from 'ml-kernel'
import KOPLS from 'ml-pls'

var kernel = new Kernel('gaussian', {
    sigma: 25
});

var cls = new KOPLS({
    orthogonalComponents: 10,
    predictiveComponents: 1,
    kernel: kernel
});

cls.train(Xtrain, Ytrain);
var {
    prediction, // prediction
    predScoreMat, // Score matrix over prediction
    predYOrthVectors // Y-Orthogonal vectors over prediction
} = cls.predict(Xtest)
```

## [API Documentation](./docs/index.html)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pls.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-pls
[travis-image]: https://img.shields.io/travis/mljs/pls/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/pls
[david-image]: https://img.shields.io/david/mljs/pls.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/pls
[download-image]: https://img.shields.io/npm/dm/ml-pls.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-pls
