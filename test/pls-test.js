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

        (result[0][0] > result[0][1]).should.be.ok;
        (result[1][0] > result[1][1]).should.be.ok;
        (result[2][0] > result[2][1]).should.be.ok;
        (result[3][0] < result[3][1]).should.be.ok;
    });

    it("Export and import", function () {
        var model = pls.export();
        var newpls = PLS.load(model);
        var result = newpls.predict(training);

        (result[0][0] > result[0][1]).should.be.ok;
        (result[1][0] > result[1][1]).should.be.ok;
        (result[2][0] > result[2][1]).should.be.ok;
        (result[3][0] < result[3][1]).should.be.ok;
    });
});
