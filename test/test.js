"use strict";

var PLS = require("../src/pls");
var Matrix = require("ml-matrix");

describe("PLS-DA algorithm", function () {
    it("test with a pseudo-AND operator", function () {
        var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
        var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];

        var pls = new PLS(training, predicted);
        var result = pls.predict(training);

        (result[0][0] > result[0][1]).should.be.ok;
        (result[1][0] > result[1][1]).should.be.ok;
        (result[2][0] > result[2][1]).should.be.ok;
        (result[3][0] < result[3][1]).should.be.ok;
    });
});
