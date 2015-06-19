# Partial Least Squares (PLS)

PLS regression algorithm based on the Yi Cao Matlab implementation:

[Partial Least-Squares and Discriminant Analysis](http://www.mathworks.com/matlabcentral/fileexchange/18760-partial-least-squares-and-discriminant-analysis)

## Methods

### new PLS()

Constructor that takes no arguments.

__Example__

```js
var pls = new PLS();
```

### fit(trainingSet, predictions)

Fit the PLS model to the given training set and predictions

__Arguments__

* `trainingSet` - A matrix of the training set.
* `predictions` - A matrix of predictions with the same size of rows of the trainingSet.

__Example__

```js
var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];

pls.fit(trainingSet, predictions);
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

### export()

Exports the actual PLS to an Javascript Object.

### load(model)

Returns a new PLS with the given model.

__Arguments__

* `model` - Javascript Object generated from export() function.

## Authors

- [Jefferson Hernandez](https://github.com/JeffersonH44)

## Licence

[MIT](./LICENSE)