'use strict';

module.exports = PLS;
var Matrix = require('ml-matrix');
var Stat = require('ml-stat');

/**
* Function that pow 2 each element of a Matrix or a Vector,
* used in the apply method of the Matrix object
* @param i - index i.
* @param j - index j.
* @return The Matrix object modified at the index i, j.
* */
function pow2array(i, j) {
    this[i][j] = this[i][j] * this[i][j];
    return this;
}

/**
 * Function that given vector, returns his norm
 * @param {Vector} X
 * @returns {number} Norm of the vector
 */
function norm(X) {
    return Math.sqrt(X.clone().apply(pow2array).sum());
}

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
        this.X = model.X;
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
 * T - Score matrix of X
 * P - Loading matrix of X
 * U - Score matrix of Y
 * Q - Loading matrix of Y
 * B - Matrix of regression coefficient
 * W - Weght matrix of X
 *
 * @param {Matrix} trainingSet - Dataset to be apply the model
 * @param {Matrix} predictions - Predictions over each case of the dataset
 */
PLS.prototype.fit = function (trainingSet, predictions) {
    if(trainingSet.length !== predictions.length)
        throw new RangeError("The number of predictions and elements in the dataset must be the same");

    var tolerance = 1e-10;
    var X = featureNormalize(Matrix(trainingSet).clone()).result;
    var resultY = featureNormalize(Matrix(predictions).clone());
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

    var n = Math.max(cx, cy); // components of the pls
    var T = Matrix.zeros(rx, n);
    var P = Matrix.zeros(cx, n);
    var U = Matrix.zeros(ry, n);
    var Q = Matrix.zeros(cy, n);
    var B = Matrix.zeros(n, n);
    var W = P.clone();
    var k = 0;

    while(norm(Y) > tolerance && k < n) {
        var transposeX = X.transpose();
        var transposeY = Y.transpose();

        var tIndex = maxSumColIndex(X.clone().mulM(X));
        var uIndex = maxSumColIndex(Y.clone().mulM(Y));

        var t1 = X.getColumnVector(tIndex);
        var u = Y.getColumnVector(uIndex);
        var t = Matrix.zeros(rx, 1);

        while(norm(t1.clone().sub(t)) > tolerance) {
            var w = transposeX.mmul(u);
            w.div(norm(w));
            t = t1;
            t1 = X.mmul(w);
            var q = transposeY.mmul(t1);
            q.div(norm(q));
            u = Y.mmul(q);
        }

        t = t1;
        var num = transposeX.mmul(t);
        var den = (t.transpose().mmul(t))[0][0];
        var p = num.div(den);
        var pnorm = norm(p);
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

    this.X = X;
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
        X: this.X,
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

/**
 * Function that predict the behavior of the given dataset.
 * @param dataset - data to be predicted.
 * @returns {Matrix} - predictions of each element of the dataset.
 */
PLS.prototype.predict = function (dataset) {
    var X = Matrix(dataset).clone();
    var normalization = featureNormalize(X);
    X = normalization.result;
    var means = normalization.means;
    var std = normalization.std;
    var Y = X.mmul(this.PBQ);
    Y.mulRowVector(this.ystd);
    // be careful because its suposed to be a sumRowVector but the mean
    // is negative here in the case of the and
    Y.subRowVector(this.ymean);
    return Y;
};

PLS.prototype.applyOSC = function () {
    var orthoP = new Matrix(this.P.rows, this.P.columns);
    var orthoW = new Matrix(this.W.rows, this.W.columns);
    var orthoT = new Matrix(this.T.rows, this.T.columns);
    for(var i = 0; i < 1; ++i) {
        var p = this.P.getColumnVector(i);
        var w = this.W.getColumnVector(i);
        var t = this.T.getColumnVector(i);
        var wTranspose = w.transpose();
        var tTranspose = t.transpose();

        var numerator = wTranspose.clone().mmul(p);
        var denominator = wTranspose.clone().mmul(w);
        this.orthoW = p.clone().sub(w.mulS(numerator.div(denominator)[0][0]));
        this.orthoT = this.X.clone().mmul(this.orthoW);

        numerator = this.X.transpose().mmul(this.orthoT);
        denominator = this.orthoT.transpose().mmul(this.orthoT);
        this.orthoP = numerator.divS(denominator[0][0]);
    }

    this.OSC = true;

    return this.X.clone().sub(this.orthoT.clone().mmul(this.orthoP.transpose()));
};

/**
 * Function that normalize the dataset and return the means and
 * standard deviation of each feature.
 * @param dataset
 * @returns {{result: Matrix, means: (*|number), std: Matrix}} dataset normalized, means
 *                                                             and standard deviations
 */
function featureNormalize(dataset) {
    var means = Stat.matrix.mean(dataset);
    var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
    means = Matrix.rowVector(means);

    var result = dataset.addRowVector(means.neg());
    return {result: result.divRowVector(std), means: means, std: std};
}
