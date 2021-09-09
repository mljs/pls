/* eslint-disable import/no-unresolved */
/* eslint-disable no-console */
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