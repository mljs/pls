'use strict';

module.exports = PLS;
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');
var Utils = require('./utils');

/**
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} X
 * @returns {number} index of the maximum
 */
function maxSumColIndex(X) {
    var maxIndex = 0;
    var maxSum = -Infinity;
    for(var i = 0; i < X.columns; ++i) {
        var currentSum = X.getColumnVector(i).sum();
        if(currentSum > maxSum) {
            maxSum = currentSum;
            maxIndex = i;
        }
    }
    return maxIndex;
}

/**
 * Constructor of the PLS model.
 * @param reload - used for load purposes.
 * @param model - used for load purposes.
 * @constructor
 */
function PLS(reload, model) {
    if(reload) {
        this.E = model.E;
        this.F = model.F;
        this.ymean = model.ymean;
        this.ystd = model.ystd;
        this.PBQ = model.PBQ;
        this.T = model.T;
        this.P = model.P;
        this.U = model.U;
        this.Q = model.Q;
        this.W = model.W;
        this.B = model.B;
        this.OSC = model.OSC;
        this.orthoP = model.orthoP;
        this.orthoW = model.orthoW;
        this.orthoT = model.orthoT;
    }
}

/**
 * Function that fit the model with the given data and predictions, in this function is calculated the
 * following outputs:
 *
 * T - Score matrix of E
 * P - Loading matrix of E
 * U - Score matrix of F
 * Q - Loading matrix of F
 * B - Matrix of regression coefficient
 * W - Weight matrix of E
 *
 * @param {Matrix} trainingSet - Dataset to be apply the model
 * @param {Matrix} predictions - Predictions over each case of the dataset
 */
PLS.prototype.fit = function (trainingSet, predictions, latentVectors, tolerance) {
    if(trainingSet.length !== predictions.length)
        throw new RangeError("The number of predictions and elements in the dataset must be the same");

    //var tolerance = 1e-9;
    var X = Utils.featureNormalize(Matrix(trainingSet).clone()).result;
    var resultY = Utils.featureNormalize(Matrix(predictions).clone());
    this.ymean = resultY.means;
    this.ystd = resultY.std;
    var Y = resultY.result;

    var rx = X.rows;
    var cx = X.columns;
    var ry = Y.rows;
    var cy = Y.columns;

    if(rx != ry) {
        throw new RangeError("dataset cases is not the same as the predictions");
    }

    var n = latentVectors; //Math.max(cx, cy); // components of the pls
    var T = Matrix.zeros(rx, n);
    var P = Matrix.zeros(cx, n);
    var U = Matrix.zeros(ry, n);
    var Q = Matrix.zeros(cy, n);
    var B = Matrix.zeros(n, n);
    var W = P.clone();
    var k = 0;

    while(Utils.norm(Y) > tolerance && k < n) {
        var transposeX = X.transpose();
        var transposeY = Y.transpose();

        var tIndex = maxSumColIndex(X.clone().mulM(X));
        var uIndex = maxSumColIndex(Y.clone().mulM(Y));

        var t1 = X.getColumnVector(tIndex);
        var u = Y.getColumnVector(uIndex);
        var t = Matrix.zeros(rx, 1);

        while(Utils.norm(t1.clone().sub(t)) > tolerance) {
            var w = transposeX.mmul(u);
            w.div(Utils.norm(w));
            t = t1;
            t1 = X.mmul(w);
            var q = transposeY.mmul(t1);
            q.div(Utils.norm(q));
            u = Y.mmul(q);
        }

        t = t1;
        var num = transposeX.mmul(t);
        var den = (t.transpose().mmul(t))[0][0];
        var p = num.div(den);
        var pnorm = Utils.norm(p);
        p.div(pnorm);
        t.mul(pnorm);
        w.mul(pnorm);

        num = u.transpose().mmul(t);
        den = (t.transpose().mmul(t))[0][0];
        var b = (num.div(den))[0][0];
        X.sub(t.mmul(p.transpose()));
        Y.sub(t.clone().mul(b).mmul(q.transpose()));

        T.setColumn(k, t);
        P.setColumn(k, p);
        U.setColumn(k, u);
        Q.setColumn(k, q);
        W.setColumn(k, w);
        B[k][k] = b;
        k++;
    }

    k--;
    n--;
    T = T.subMatrix(0, T.rows - 1, 0, k);
    P = P.subMatrix(0, P.rows - 1, 0, k);
    U = U.subMatrix(0, U.rows - 1, 0, k);
    Q = Q.subMatrix(0, Q.rows - 1, 0, k);
    W = W.subMatrix(0, W.rows - 1, 0, k);
    B = B.subMatrix(0, k, 0, k);

    this.E = X;
    this.F = Y;
    this.T = T;
    this.P = P;
    this.U = U;
    this.Q = Q;
    this.W = W;
    this.B = B;
    this.PBQ = P.mmul(B).mmul(Q.transpose());
    this.OSC = false;
    this.orthoW = undefined;
    this.orthoT = undefined;
    this.orthoP = undefined;
};

/**
 * Function that predict the behavior of the given dataset.
 * @param dataset - data to be predicted.
 * @returns {Matrix} - predictions of each element of the dataset.
 */
PLS.prototype.predict = function (dataset) {
    var X = Matrix(dataset).clone();
    var normalization = Utils.featureNormalize(X);
    X = normalization.result;
    var means = normalization.means;
    var std = normalization.std;
    var Y = X.mmul(this.PBQ).add(this.F);
    Y.mulRowVector(this.ystd);
    // be careful because its suposed to be a sumRowVector but the mean
    // is negative here in the case of the and
    Y.subRowVector(this.ymean);
    return Y;
};

PLS.prototype.applyOSC = function (trainingSet) {
    var X = Matrix(trainingSet).clone();
    var w = this.W.getColumnVector(0);
    var p = this.P.getColumnVector(0);

    var numerator = w.transpose().mmul(p);
    var denominator = w.transpose().mmul(w);
    var resultDivision = numerator.div(denominator)[0][0];
    this.orthoW = p.sub(w.mulS(resultDivision));

    this.orthoT = X.mmul(this.orthoW);

    numerator = X.transpose().mmul(this.orthoT);
    denominator = this.orthoT.transpose().mmul(this.orthoT)[0][0];
    this.orthoP = numerator.divS(denominator);

    return X.sub(this.orthoT.mmul(this.orthoP.transpose()));
};

/**
 * Load a PLS model from an Object
 * @param model
 * @returns {PLS} - PLS object from the given model
 */
PLS.load = function (model) {
    if(model.modelName !== 'PLS')
        throw new RangeError("The current model is invalid!");

    return new PLS(true, model);
};

/**
 * Function that exports a PLS model to an Object.
 * @returns {{modelName: string, ymean: *, ystd: *, PBQ: *}} model.
 */
PLS.prototype.export = function () {
    return {
        modelName: "PLS",
        E: this.E,
        F: this.F,
        ymean: this.ymean,
        ystd: this.ystd,
        PBQ: this.PBQ,
        T: this.T,
        P: this.P,
        U: this.U,
        Q: this.Q,
        W: this.W,
        B: this.B,
        OSC: this.OSC,
        orthoW: this.orthoW,
        orthoT: this.orthoT,
        orthoP: this.orthoP
    };
};
