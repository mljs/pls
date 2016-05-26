'use strict';

var Matrix = require('ml-matrix');
var Utils = require('./utils');

class PLS {
    constructor(reload, model) {
        if (reload) {
            this.ymean = Matrix.checkMatrix(model.ymean);
            this.ystd = Matrix.checkMatrix(model.ystd);
            this.PBQ = Matrix.checkMatrix(model.PBQ);
            this.R2X = model.R2X;
        } else {
            this.ymean = null;
            this.ystd = null;
            this.PBQ = null;
            this.R2X = 0;
        }
    }

    /**
     * Fits the model with the given data and predictions, in this function is calculated the
     * following outputs:
     *
     * T - Score matrix of X
     * P - Loading matrix of X
     * U - Score matrix of Y
     * Q - Loading matrix of Y
     * B - Matrix of regression coefficient
     * W - Weight matrix of X
     *
     * @param {Matrix} trainingSet - Dataset to be apply the model
     * @param {Matrix} predictions - Predictions over each case of the dataset
     * @param {Object} options - recieves the latentVectors and the tolerance of each step of the PLS
     */
    train(trainingSet, predictions, options) {
        if(options === undefined) options = {};

        var latentVectors = options.latentVectors;
        if (latentVectors === undefined || isNaN(latentVectors)) {
            throw new RangeError('Latent vector must be a number.');
        }

        var tolerance = options.tolerance;
        if (tolerance === undefined || isNaN(tolerance)) {
            throw new RangeError('Tolerance must be a number');
        }

        if (trainingSet.length !== predictions.length)
            throw new RangeError('The number of predictions and elements in the dataset must be the same');

        //var tolerance = 1e-9;
        var X = Utils.featureNormalize(new Matrix(trainingSet)).result;
        var resultY = Utils.featureNormalize(new Matrix(predictions));
        this.ymean = resultY.means.neg();
        this.ystd = resultY.std;
        var Y = resultY.result;

        var rx = X.rows;
        var cx = X.columns;
        var ry = Y.rows;
        var cy = Y.columns;

        if(rx != ry) {
            throw new RangeError('dataset cases is not the same as the predictions');
        }

        var ssqXcal = X.clone().mul(X).sum(); // for the r²
        var sumOfSquaresY = Y.clone().mul(Y).sum();

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
        T = T.subMatrix(0, T.rows - 1, 0, k);
        P = P.subMatrix(0, P.rows - 1, 0, k);
        U = U.subMatrix(0, U.rows - 1, 0, k);
        Q = Q.subMatrix(0, Q.rows - 1, 0, k);
        W = W.subMatrix(0, W.rows - 1, 0, k);
        B = B.subMatrix(0, k, 0, k);

        // TODO: review of R2Y
        //this.R2Y = t.transpose().mmul(t).mul(q[k][0]*q[k][0]).divS(ssqYcal)[0][0];

        this.ssqYcal = sumOfSquaresY;
        this.E = X;
        this.F = Y;
        this.T = T;
        this.P = P;
        this.U = U;
        this.Q = Q;
        this.W = W;
        this.B = B;
        this.PBQ = P.mmul(B).mmul(Q.transpose());
        this.R2X = t.transpose().mmul(t).mmul(p.transpose().mmul(p)).div(ssqXcal)[0][0];
    }

    /**
     * Predicts the behavior of the given dataset.
     * @param dataset - data to be predicted.
     * @returns {Matrix} - predictions of each element of the dataset.
     */
    predict(dataset) {
        var X = Matrix.checkMatrix(dataset);
        var normalization = Utils.featureNormalize(X);
        X = normalization.result;
        var Y = X.mmul(this.PBQ);
        Y.mulRowVector(this.ystd);
        Y.addRowVector(this.ymean);
        return Y;
    }

    /**
     * Returns the explained variance on training of the PLS model
     * @return {number}
     */
    getExplainedVariance() {
        return this.R2X;
    }
    
    toJSON() {
        return {
            name: 'PLS',
            R2X: this.R2X,
            ymean: this.ymean,
            ystd: this.ystd,
            PBQ: this.PBQ,
        };
    }

    /**
     * Load a PLS model from a JSON Object
     * @param model
     * @return {PLS} - PLS object from the given model
     */
    static load(model) {
        if (model.name !== 'PLS')
            throw new RangeError('Invalid model: ' + model.name);
        return new PLS(true, model);
    }
}

module.exports = PLS;

/**
 * Retrieves the sum at the column of the given matrix.
 * @param matrix
 * @param column
 * @returns {number}
 */
function getColSum(matrix, column) {
    var sum = 0;
    for (var i = 0; i < matrix.rows; i++) {
        sum += matrix[i][column];
    }
    return sum;
}

/**
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} data
 * @returns {number} index of the maximum
 */
function maxSumColIndex(data) {
    var maxIndex = 0;
    var maxSum = -Infinity;
    for(var i = 0; i < data.columns; ++i) {
        var currentSum = getColSum(data, i);
        if(currentSum > maxSum) {
            maxSum = currentSum;
            maxIndex = i;
        }
    }
    return maxIndex;
}
