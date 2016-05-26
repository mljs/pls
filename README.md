# Partial Least Squares (PLS)

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

PLS regression algorithm based on the Yi Cao Matlab implementation:

[Partial Least-Squares and Discriminant Analysis](http://www.mathworks.com/matlabcentral/fileexchange/18760-partial-least-squares-and-discriminant-analysis)

## installation

`$ npm install ml-pls`

## Methods

### new PLS()

Constructor that takes no arguments.

__Example__

```js
var pls = new PLS();
```

### train(trainingSet, predictions, options)

Train the PLS model to the given training set and predictions

__Arguments__

* `trainingSet` - A matrix of the training set.
* `predictions` - A matrix of predictions with the same size of rows of the trainingSet.
* `options` - A Javascript object with to values, latentVectors and the tolerance of each step of the PLS algorithm

__Example__

```js
var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];
var options = {
  latentVectors: 10,
  tolerance: 1e-4
};

pls.train(trainingSet, predictions, options);
```

### predict(dataset)

Predict the values of the dataset.

__Arguments__

* `dataset` - A matrix that contains the dataset.

__Example__

```js
var dataset = [[0, 0], [0, 1], [1, 0], [1, 1]];

var ans = pls.predict(dataset);
```

### getExplainedVariance()

Returns the explained variance on training

### export()

Exports the actual PLS to an Javascript Object.

### load(model)

Returns a new PLS with the given model.

__Arguments__

* `model` - Javascript Object generated from export() function.

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
