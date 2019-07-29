
import { getNumbers, getClasses } from 'ml-dataset-iris';

import { getFolds } from '../utils';

const iris = getNumbers();
const metadata = getClasses();


describe('utils', () => {
  it('test getFolds', () => {
    let folds = getFolds(metadata, 5);
    expect(folds.length).toStrictEqual(5);
    expect(folds[0].testIndex.length).toStrictEqual(30);
    expect(folds[0].trainIndex.length).toStrictEqual(120);
  });
});
