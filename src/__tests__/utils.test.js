import Matrix from 'ml-matrix';

import { Q2 } from '../utils';

describe.skip('utils', () => {
  it('test Q2', () => {
    let t = Q2([[1], [2]], [[1], [2]]);
    console.log(t);
    expect(t).toStrictEqual(1);
  });
});
