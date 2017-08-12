import {PLS} from '..'

describe("PLS-DA algorithm", function () {
    var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
    var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];
    var pls = new PLS(training, predicted);
    pls.train({
        latentVectors: 2,
        tolerance: 1e-5
    });

    test("test with a pseudo-AND operator", function () {
        var result = pls.predict(training);

        expect(result[0][0]).toBeGreaterThan(result[0][1]);
        expect(result[1][0]).toBeGreaterThan(result[1][1]);
        expect(result[2][0]).toBeGreaterThan(result[2][1]);
        expect(result[3][0]).toBeLessThan(result[3][1]);
    });

    test('Random points test', function () {
        var training = [[0.323, 34, 56, 23], [2.23, 43, 32, 83]];
        var predicted = [[23], [15]];

        var newPls = new PLS(training, predicted);
        newPls.train({
            latentVectors: 2,
            tolerance: 1e-5
        });
        var result = newPls.predict(training);

        expect(result[0][0]).toEqual(predicted[0][0]);
        expect(result[1][0]).toEqual(predicted[1][0]);
    });

    test("Export and import", function () {
        var model = JSON.parse(JSON.stringify(pls.toJSON()));

        var properties = ['name', 'R2X', 'meanX', 'stdDevX', 'meanY', 'stdDevY', 'PBQ'];
        for (var prop of properties) {
            expect(model).toHaveProperty(prop);
        }

        var newpls = PLS.load(model);
        var result = newpls.predict(training);

        expect(result[0][0]).toBeGreaterThan(result[0][1]);
        expect(result[1][0]).toBeGreaterThan(result[1][1]);
        expect(result[2][0]).toBeGreaterThan(result[2][1]);
        expect(result[3][0]).toBeLessThan(result[3][1]);
    });

    /*
    * Test case based on the following document:
    *
    * Partial Least Squares (PLS) regression by Herve Abdi
    * https://www.utdallas.edu/~herve/Abdi-PLS-pretty.pdf
    *
    * */
    test('Wine test with getExplainedVariance', function () {
        var dataset = [[7, 7, 13, 7],
                       [4, 3, 14, 7],
                       [10, 5, 12, 5],
                       [16, 7, 11, 3],
                       [13, 3, 10, 3]];
        var predictions = [[14, 7, 8],
                           [10, 7, 6],
                           [8, 5, 5],
                           [2, 4, 7],
                           [6, 2, 4]];

        var winePLS = new PLS(dataset, predictions);
        var latentStructures = 3;
        var tolerance = 1e-5;
        winePLS.train({
            latentVectors: latentStructures,
            tolerance: tolerance
        });
        var result = winePLS.predict(dataset);

        expect(result[2][0]).toBeCloseTo(predictions[2][0], -1);
        expect(result[2][1]).toBeCloseTo(predictions[2][1], -1);
        expect(result[2][2]).toBeCloseTo(predictions[2][2], -1);
        expect(winePLS.getExplainedVariance()).toBeCloseTo(0.02, 1);
    });
});

/*describe('OPLS', function () {
    var X0 = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
    var X1 = [[-2.18, -2.18], [1.84, -0.16], [-0.48, 1.52], [0.83, 0.83]];
    var y = [[2], [2], [0], [4]];

    test('Main test', function () {
        var opls = new OPLS(X1, y, 1);
        expect(opls.R2X).toBeCloseTo(0.7402, 1);
    });
});*/
