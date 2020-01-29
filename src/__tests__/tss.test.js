import Matrix from 'ml-matrix';

import { tss } from '../tss.js';

describe('tss', () => {
  let x = Matrix.from1DArray(1, 2, [1, 2]);
  it('1+1=2', () => {
    let t = tss(x);
    expect(t).toStrictEqual(5);
  });
});
