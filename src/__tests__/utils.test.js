import Matrix from 'ml-matrix';

import { Q2, getFolds } from '../utils';

describe('utils', () => {
  it('test getFolds', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let folds = getFolds(metadata, 5);
    expect(folds.length).toStrictEqual(5);
    expect(folds[0].testIndex.length).toStrictEqual(30);
    expect(folds[0].trainIndex.length).toStrictEqual(120);
  });
});
