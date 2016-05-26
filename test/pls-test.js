"use strict";

var PLS = require("..");
var OPLS = require("..").OPLS;
var Matrix = require("ml-matrix");

describe("PLS-DA algorithm", function () {
    var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
    var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];
    var pls = new PLS();
    pls.train(training, predicted, {
        latentVectors: 2,
        tolerance: 1e-5
    });

    it("test with a pseudo-AND operator", function () {
        var result = pls.predict(training);

        (result[0][0]).should.be.greaterThan(result[0][1]);
        (result[1][0]).should.be.greaterThan(result[1][1]);
        (result[2][0]).should.be.greaterThan(result[2][1]);
        (result[3][0]).should.be.lessThan(result[3][1]);
    });

    it('Random points test', function () {
        var training = [[0.323, 34, 56, 23], [2.23, 43, 32, 83]];
        var predicted = [[23], [15]];

        var newPls = new PLS();
        newPls.train(training, predicted,{
            latentVectors: 2,
            tolerance: 1e-5
        });
        var result = newPls.predict(training);

        result[0][0].should.be.equal(predicted[0][0]);
        result[1][0].should.be.equal(predicted[1][0]);
    });

    it("Export and import", function () {
        var model = JSON.parse(JSON.stringify(pls.toJSON()));

        model.should.have.properties(['name', 'R2X', 'ymean', 'ystd', 'PBQ']);

        var newpls = PLS.load(model);
        var result = newpls.predict(training);

        (result[0][0]).should.be.greaterThan(result[0][1]);
        (result[1][0]).should.be.greaterThan(result[1][1]);
        (result[2][0]).should.be.greaterThan(result[2][1]);
        (result[3][0]).should.be.lessThan(result[3][1]);
    });

    /*
    * Test case based on the following document:
    *
    * Partial Least Squares (PLS) regression by Herve Abdi
    * https://www.utdallas.edu/~herve/Abdi-PLS-pretty.pdf
    *
    * */
    it('Wine test with getExplainedVariance', function () {
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

        var winePLS = new PLS();
        var latentStructures = 3;
        var tolerance = 1e-5;
        winePLS.train(dataset, predictions, {
            latentVectors: latentStructures,
            tolerance: tolerance
        });
        dataset.pop();
        var result = winePLS.predict(dataset);

        result[2][0].should.be.approximately(predictions[2][0], 1);
        result[2][1].should.be.approximately(predictions[2][1], 1);
        result[2][2].should.be.approximately(predictions[2][2], 1);
        winePLS.getExplainedVariance().should.be.approximately(0.02, 1e-2);
    });
});

describe('OPLS', function () {
    var X0 = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
    var X1 = [[-2.18, -2.18], [1.84, -0.16], [-0.48, 1.52], [0.83, 0.83]];
    var y = [[2], [2], [0], [4]];

    it('Main test', function () {
        var opls = new OPLS(X1, y, 1);
        opls.R2X.should.be.approximately(0.7402, 1e-1);
    });
});
