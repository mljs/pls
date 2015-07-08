"use strict";

var PLS = require("..");
var Matrix = require("ml-matrix");

describe("PLS-DA algorithm", function () {
    var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
    var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];
    var pls = new PLS();
    pls.fit(training, predicted);

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
        newPls.fit(training, predicted);
        var result = newPls.predict(training);

        result[0][0].should.be.equal(predicted[0][0]);
        result[1][0].should.be.equal(predicted[1][0]);
    });

    it("Export and import", function () {
        var model = pls.export();
        var newpls = PLS.load(model);
        var result = newpls.predict(training);

        (result[0][0]).should.be.greaterThan(result[0][1]);
        (result[1][0]).should.be.greaterThan(result[1][1]);
        (result[2][0]).should.be.greaterThan(result[2][1]);
        (result[3][0]).should.be.lessThan(result[3][1]);
    });
});
