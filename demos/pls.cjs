/* eslint-disable import/no-unresolved */
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
