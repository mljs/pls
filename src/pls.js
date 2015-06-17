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
    for(var i = 0; i < X.column; ++i) {
        var currentSum = X.getColumnVector(i).sum();
        if(currentSum > maxSum) {
            maxSum = currentSum;
            maxIndex = i;
        }
    }
    return i;
}

/**
 * Construction of the PLS model that takes
 * @param {Matrix} dataset - Dataset to be apply the model
 * @param {Matrix} predictions - Predictions over each case of the dataset
 * @param reload - used internally for load purposes.
 * @constructor
 * @throws RangeError - if the number of elements in predictions isn't be
 *                      the same to the dataset
 */
function PLS(dataset, predictions, reload) {
    if(reload) {
        this.ymean = dataset.ymean;
        this.ystd = dataset.ystd;
        this.PBQ = dataset.PBQ;
    } else {
        if(dataset.length !== predictions.length)
            throw new RangeError("The number of predictions and elements in the dataset must be the same");

        var tolerance = 1e-10;
        var X = featureNormalize(Matrix(dataset).clone()).result;
        var resultY = featureNormalize(Matrix(predictions).clone());
        this.ymean = resultY.means;
        this.ystd = resultY.std;
        var Y = resultY.result;

        var rx = X.rows;
        var cx = X.columns;
        var ry = Y.rows;
        var cy = Y.columns;

        if(rx != ry) {
            throw new Error("dataset cases is not the same as the predictions");
        }

        var n = Math.max(cx, cy);
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
                var u = Y.mmul(q);
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
            var b = (num.div(den))[0][0];
            X.sub(t.mmul(p.transpose()));
            Y.sub(t.mmul(q.transpose()).mul(b));


            T.addColumn(k, t);
            P.addColumn(k, p);
            U.addColumn(k, u);
            Q.addColumn(k, q);
            W.addColumn(k, w);
            B[k][k] = b;
            k++;
        }

        // NOTE: some variables commented because maybe it's needed
        // in the future

        k--;
        n--;
        //T = T.subMatrix(0, n, 0, k);
        P = P.subMatrix(0, n, 0, k);
        //U = U.subMatrix(0, n, 0, k);
        Q = Q.subMatrix(0, n, 0, k);
        //W = W.subMatrix(0, n, 0, k);
        B = B.subMatrix(0, k, 0, k);

        // this.T = T;
        // this.P = P;
        // this.U = U;
        // this.Q = Q;
        // this.W = W;
        // this.B = B;
        this.PBQ = P.mmul(B).mmul(Q.transpose());
    }
}

/**
 * Load a PLS model from an Object
 * @param model
 * @returns {PLS} - PLS object from the given model
 */
PLS.load = function (model) {
    if(model.modelName !== 'PLS')
        throw new RangeError("The current model is invalid!");

    return new PLS(model, null, true);
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

/**
 * Function that normalize the dataset and return the means and
 * standard deviation of each feature.
 * @param dataset
 * @returns {{result: Matrix, means: (*|number), std: Matrix}}
 */
function featureNormalize(dataset) {
    var means = Stat.matrix.mean(dataset);
    var std = Matrix.rowVector(Stat.matrix.standardDeviation(dataset, means, true));
    means = Matrix.rowVector(means);

    var result = dataset.addRowVector(means.neg());
    return {result: result.divRowVector(std), means: means, std: std};
}
