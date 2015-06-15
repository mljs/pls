"use strict";

var PLS = require("../src/pls");
var Matrix = require("ml-matrix");

describe("Main test", function () {
    it("PLS-DA algorithm", function () {
        var training = [[0.1, 0.02], [0.25, 1.01] ,[0.95, 0.01], [1.01, 0.96]];
        var predicted = [[1, 0], [1, 0], [1, 0], [0, 1]];

        var pls = new PLS(training, predicted);
        console.log(pls.predict(training));
    });
});
