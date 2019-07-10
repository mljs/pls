import { Matrix } from 'ml-matrix';
import { CV } from 'ml-cross-validation';

import { OPLS } from '../OPLS.js';
import { sampleAClass, getTrainTest } from '../utils.js';
import { METADATA } from '../METADATA.js';

describe('OPLS', () => {
  it('test LABELS', () => {
    let rawData = require('../../data/irisDataset.json');
    let metadata = rawData.map((d) => d[4]);
    let L = new METADATA().addMetadata('iris', metadata);
    expect(L.getMetadata('iris').nClass).toStrictEqual(3);
    expect(L.getMetadata('iris').classMatrix.rows).toStrictEqual(150);
  });
});
