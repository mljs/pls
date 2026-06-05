import { Matrix } from 'ml-matrix';
import { expect, test } from 'vitest';

import { tss } from '../../util/tss.js';

test('1+1=2', () => {
  let x = Matrix.from1DArray(1, 2, [1, 2]);
  let t = tss(x);

  expect(t).toBe(5);
});
