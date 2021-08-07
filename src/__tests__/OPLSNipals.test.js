import { toBeDeepCloseTo } from 'jest-matcher-deep-close-to';
import { getNumbers, getClasses } from 'ml-dataset-iris';
import { METADATA } from 'ml-dataset-metadata';
import { Matrix } from 'ml-matrix';

import { OPLSNipals } from '../OPLSNipals';

expect.extend({ toBeDeepCloseTo });

const iris = getNumbers();
const metadata = getClasses();
const newM = new METADATA([metadata], { headers: ['iris'] });
let numericValues = newM.get('iris', { format: 'matrix' }).values;

describe('opls-nipals', () => {
  it('test pls-nipals iris data', () => {
    expect(iris).toHaveLength(150);
    expect(metadata).toHaveLength(150);
  });

  it('test pls-nipals dataArray', () => {
    let y = Matrix.from1DArray(150, 1, metadata);
    let x = new Matrix(iris);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = OPLSNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(150);
  });

  it('test opls-nipals with iris', () => {
    let x = new Matrix(iris);
    let y = numericValues.clone();
    let model = OPLSNipals(x, y);

    expect(model.scoresXpred.getRow(0)).toBeDeepCloseTo([5.66576130528549]);
    expect(model.scoresXpred.getRow(1)).toBeDeepCloseTo([5.357904200567914]);
    expect(model.scoresXpred.getRow(148)).toBeDeepCloseTo([9.161531522704093]);
    expect(model.scoresXpred.getRow(149)).toBeDeepCloseTo([8.539939678663997]);

    expect(model.loadingsXpred.getRow(0)).toBeDeepCloseTo([
      0.7533846125482135,
      0.37991165000540095,
      0.5175455876998517,
      0.1697637836019074,
    ]);

    expect(model.weightsXPred.getColumn(0)).toBeDeepCloseTo([
      0.7191635887943787,
      0.3280487739174021,
      0.5781304613979087,
      0.20236823883090044,
    ]);

    expect(model.scoresXOrtho.getRow(0)).toBeDeepCloseTo([2.8553426763334433]);
    expect(model.scoresXOrtho.getRow(1)).toBeDeepCloseTo([2.501798988578702]);
    expect(model.scoresXOrtho.getRow(148)).toBeDeepCloseTo([-0.14717957843937]);
    expect(model.scoresXOrtho.getRow(149)).toBeDeepCloseTo([-0.1097900904882]);

    expect(model.loadingsXOrtho.getRow(0)).toBeDeepCloseTo([
      1.893676431872939,
      1.2763813703908802,
      0.5781825736661366,
      0.10014507134716187,
    ]);

    expect(model.weightsXOrtho.getRow(0)).toBeDeepCloseTo([
      0.36913469592136616,
      0.5594334971409369,
      -0.6535157770509538,
      -0.35169712493733896,
    ]);

    expect(model.loadingsY.getRow(0)).toBeDeepCloseTo([0.14558532608468372]);
  });

  it('test opls-nipals with iris multi Y', () => {
    let x = new Matrix(iris);
    let y = new Matrix(numericValues.rows, 2);
    y.setColumn(0, numericValues);
    y.setColumn(1, numericValues);
    let model = OPLSNipals(x, y);
    expect(model.loadingsXpred.getRow(0)).toBeDeepCloseTo([
      -39.87921132989588,
      -18.15477710189015,
      -32.14670745016701,
      -11.264000394844325,
    ]);

    expect(model.weightsXPred.getColumn(0)).toBeDeepCloseTo([
      0.7191635887943787,
      0.3280487739174021,
      0.5781304613979087,
      0.20236823883090044,
    ]);

    expect(model.scoresXpred.getRow(0)).toBeDeepCloseTo([5.66576130528549]);
    expect(model.scoresXpred.getRow(1)).toBeDeepCloseTo([5.357904200567914]);
    expect(model.scoresXpred.getRow(148)).toBeDeepCloseTo([9.161531522704093]);
    expect(model.scoresXpred.getRow(149)).toBeDeepCloseTo([8.539939678663997]);

    expect(model.loadingsXOrtho.getRow(0)).toBeDeepCloseTo([
      -0.7534815884584046,
      -0.37993723094817833,
      -0.5176646384052347,
      -0.16980932815451685,
    ]);

    expect(model.weightsXOrtho.getRow(0)).toBeDeepCloseTo([
      -0.7185459890650463,
      -0.32711384788980913,
      -0.579221276691814,
      -0.20295542551201248,
    ]);

    expect(model.scoresXOrtho.getRow(0)).toBeDeepCloseTo([-5.660983884317011]);
    expect(model.scoresXOrtho.getRow(1)).toBeDeepCloseTo([-5.353717762559098]);
    expect(model.scoresXOrtho.getRow(148)).toBeDeepCloseTo([-9.16176458784206]);
    expect(model.scoresXOrtho.getRow(149)).toBeDeepCloseTo([-8.54011115620307]);

    expect(model.loadingsY.getRow(0)).toBeDeepCloseTo([
      0.14558532608468372,
      0.14558532608468372,
    ]);
  });

  it('test pls-nipals simpleDataset', () => {
    let rawData = require('../../data/simpleDataset.json');
    let x = new Matrix(rawData);
    let y = Matrix.from1DArray(8, 1, [1, 1, 2, 2, 3, 1, 3, 3]);

    x = x.center('column').scale('column');
    y = y.center('column').scale('column');

    let model = OPLSNipals(x, y);

    expect(model.scoresXOrtho.to1DArray()).toHaveLength(8);

    expect(model.weightsXPred.to1DArray()).toStrictEqual([0.5, -0.5, 0.5, 0.5]);
  });
});
