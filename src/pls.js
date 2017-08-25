import Matrix from 'ml-matrix';
import Stat from 'ml-stat/matrix';
import * as Utils from './utils';

export class PLS {

    /**
     * Constructor for Partial Least Squares (PLS)
     * @param {object} options
     * @param {number} [options.latentVectors] - Number of latent vector to get (if the algorithm doesn't find a good model below the tolerance)
     * @param {number} [options.tolerance=1e-5]
     * @param {boolean} [options.scale=true] - rescale dataset using mean.
     * @param {object} model - for load purposes.
     */
    constructor(options, model) {
        if (options === true) {
            this.meanX = model.meanX;
            this.stdDevX = model.stdDevX;
            this.meanY = model.meanY;
            this.stdDevY = model.stdDevY;
            this.PBQ = Matrix.checkMatrix(model.PBQ);
            this.R2X = model.R2X;
            this.scale = model.scale;
            this.scaleMethod = model.scaleMethod;
            this.tolerance = model.tolerance;
        } else {
            var {
                tolerance = 1e-5,
                scale = true,
            } = options;
            this.tolerance = tolerance;
            this.scale = scale;
            this.latentVectors = options.latentVectors;
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
     * @param {Matrix|Array} trainingSet
     * @param {Matrix|Array} trainingValues
     */
    train(trainingSet, trainingValues) {
        trainingSet = Matrix.checkMatrix(trainingSet);
        trainingValues = Matrix.checkMatrix(trainingValues);

        if (trainingSet.length !== trainingValues.length) {
            throw new RangeError('The number of X rows must be equal to the number of Y rows');
        }

        this.meanX = Stat.mean(trainingSet);
        this.stdDevX = Stat.standardDeviation(trainingSet, this.meanX, true);
        this.meanY = Stat.mean(trainingValues);
        this.stdDevY = Stat.standardDeviation(trainingValues, this.meanY, true);

        if (this.scale) { // here should be the ml-preprocess project
            trainingSet = trainingSet.clone().subRowVector(this.meanX).divRowVector(this.stdDevX);
            trainingValues = trainingValues.clone().subRowVector(this.meanY).divRowVector(this.stdDevY);
        }

        if (this.latentVectors === undefined) {
            this.latentVectors = Math.min(trainingSet.length - 1, trainingSet[0].length);
        }

        var rx = trainingSet.rows;
        var cx = trainingSet.columns;
        var ry = trainingValues.rows;
        var cy = trainingValues.columns;

        var ssqXcal = trainingSet.clone().mul(trainingSet).sum(); // for the r²
        var sumOfSquaresY = trainingValues.clone().mul(trainingValues).sum();

        var tolerance = this.tolerance;
        var n = this.latentVectors;
        var T = Matrix.zeros(rx, n);
        var P = Matrix.zeros(cx, n);
        var U = Matrix.zeros(ry, n);
        var Q = Matrix.zeros(cy, n);
        var B = Matrix.zeros(n, n);
        var W = P.clone();
        var k = 0;

        while (Utils.norm(trainingValues) > tolerance && k < n) {
            var transposeX = trainingSet.transpose();
            var transposeY = trainingValues.transpose();

            var tIndex = maxSumColIndex(trainingSet.clone().mulM(trainingSet));
            var uIndex = maxSumColIndex(trainingValues.clone().mulM(trainingValues));

            var t1 = trainingSet.getColumnVector(tIndex);
            var u = trainingValues.getColumnVector(uIndex);
            var t = Matrix.zeros(rx, 1);

            while (Utils.norm(t1.clone().sub(t)) > tolerance) {
                var w = transposeX.mmul(u);
                w.div(Utils.norm(w));
                t = t1;
                t1 = trainingSet.mmul(w);
                var q = transposeY.mmul(t1);
                q.div(Utils.norm(q));
                u = trainingValues.mmul(q);
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
            trainingSet.sub(t.mmul(p.transpose()));
            trainingValues.sub(t.clone().mul(b).mmul(q.transpose()));

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
        //
        this.ssqYcal = sumOfSquaresY;
        this.E = trainingSet;
        this.F = trainingValues;
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
     * @param {Matrix|Array} dataset - data to be predicted.
     * @return {Matrix} - predictions of each element of the dataset.
     */
    predict(dataset) {
        var X = Matrix.checkMatrix(dataset);
        if (this.scale) {
            X = X.subRowVector(this.meanX).divRowVector(this.stdDevX);
        }
        var Y = X.mmul(this.PBQ);
        Y = Y.mulRowVector(this.stdDevY).addRowVector(this.meanY);
        return Y;
    }

    /**
     * Returns the explained variance on training of the PLS model
     * @return {number}
     */
    getExplainedVariance() {
        return this.R2X;
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        return {
            name: 'PLS',
            R2X: this.R2X,
            meanX: this.meanX,
            stdDevX: this.stdDevX,
            meanY: this.meanY,
            stdDevY: this.stdDevY,
            PBQ: this.PBQ,
            tolerance: this.tolerance,
            scale: this.scale,
        };
    }

    /**
     * Load a PLS model from a JSON Object
     * @param {object} model
     * @return {PLS} - PLS object from the given model
     */
    static load(model) {
        if (model.name !== 'PLS') {
            throw new RangeError('Invalid model: ' + model.name);
        }
        return new PLS(true, model);
    }
}

/**
 * @private
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} data
 * @return {number} index of the maximum
 */
function maxSumColIndex(data) {
    return data.sum('column').maxIndex()[0];
}
