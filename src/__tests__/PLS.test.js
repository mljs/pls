import { PLS } from '../PLS';

describe('PLS-DA algorithm', () => {
  it('test with a pseudo-AND operator', () => {
    let training = [
      [0.1, 0.02],
      [0.25, 1.01],
      [0.95, 0.01],
      [1.01, 0.96],
    ];
    let predicted = [
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 1],
    ];

    let pls = new PLS({ latentVectors: 2 });
    pls.train(training, predicted);

    let result = pls.predict(training);

    expect(result.get(0, 0)).toBeGreaterThan(result.get(0, 1));
    expect(result.get(1, 0)).toBeGreaterThan(result.get(1, 1));
    expect(result.get(2, 0)).toBeGreaterThan(result.get(2, 1));
    expect(result.get(3, 0)).toBeLessThan(result.get(3, 1));
  });

  it('Random points test', function() {
    let training = [
      [0.323, 34, 56, 23],
      [2.23, 43, 32, 83],
    ];
    let predicted = [[23], [15]];

    let newPls = new PLS({ latentVectors: 2 });
    newPls.train(training, predicted);
    let result = newPls.predict(training);

    expect(result.get(0, 0)).toStrictEqual(predicted[0][0]);
    expect(result.get(1, 0)).toStrictEqual(predicted[1][0]);
  });

  it('Export and import', () => {
    let training = [
      [0.1, 0.02],
      [0.25, 1.01],
      [0.95, 0.01],
      [1.01, 0.96],
    ];
    let predicted = [
      [1, 0],
      [1, 0],
      [1, 0],
      [0, 1],
    ];

    let pls = new PLS({ latentVectors: 2 });
    pls.train(training, predicted);

    let model = JSON.parse(JSON.stringify(pls.toJSON()));

    let properties = [
      'name',
      'R2X',
      'meanX',
      'stdDevX',
      'meanY',
      'stdDevY',
      'PBQ',
    ];
    for (let prop of properties) {
      expect(model).toHaveProperty(prop);
    }

    let newpls = PLS.load(model);
    let result = newpls.predict(training);

    expect(result.get(0, 0)).toBeGreaterThan(result.get(0, 1));
    expect(result.get(1, 0)).toBeGreaterThan(result.get(1, 1));
    expect(result.get(2, 0)).toBeGreaterThan(result.get(2, 1));
    expect(result.get(3, 0)).toBeLessThan(result.get(3, 1));
  });

  /*
   * Test case based on the following document:
   *
   * Partial Least Squares (PLS) regression by Herve Abdi
   * https://www.utdallas.edu/~herve/Abdi-PLS-pretty.pdf
   *
   * */
  it('Wine test with getExplainedVariance', () => {
    let dataset = [
      [7, 7, 13, 7],
      [4, 3, 14, 7],
      [10, 5, 12, 5],
      [16, 7, 11, 3],
      [13, 3, 10, 3],
    ];
    let predictions = [
      [14, 7, 8],
      [10, 7, 6],
      [8, 5, 5],
      [2, 4, 7],
      [6, 2, 4],
    ];

    let winePLS = new PLS({ latentVectors: 3 });
    winePLS.train(dataset, predictions);
    let result = winePLS.predict(dataset);

    expect(result.get(2, 0)).toBeCloseTo(predictions[2][0], -1);
    expect(result.get(2, 1)).toBeCloseTo(predictions[2][1], -1);
    expect(result.get(2, 2)).toBeCloseTo(predictions[2][2], -1);
    expect(winePLS.getExplainedVariance()).toBeCloseTo(0.02, 1);
  });
});
