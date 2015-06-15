'use strict';

module.exports = PLS;
var Matrix = require('ml-matrix');

function pow2array(i, j) {
    this[i][j] = this[i][j] * this[i][j];
    return this;
}

function norm(X) {
    return Math.sqrt(X.clone().apply(pow2array).sum());
}

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

function PLS(dataset, predictions, reload) {
    if(reload) {
        // TODO: reload PLS
    } else {
        var tolerance = 1e-7;
        var X = Matrix(dataset).clone();
        var Y = Matrix(predictions).clone();

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

            var tIndex = maxSumColIndex(X.clone().mulM(transposeX));
            var uIndex = maxSumColIndex(Y.clone().mulM(transposeY));

            var t1 = X.getRowVector(tIndex);
            var u = Y.getRowVector(uIndex);
            var t = Matrix.zeros(rx, 1);

            while(norm(t1.clone().sub(t)) > tolerance) {
                var w = transposeX.mmul(u);
                w.div(norm(w));
                t = t1;
                t1 = X.clone().mmul(w);
                var q = transposeY.mmul(t1);
                q.div(norm(q));
                var u = Y.mmul(q);
            }

            t = t1;
            var num = transposeX.mmul(t);
            var den = t.transpose().mmul(t);
            var p = num.div(den);
            var pnorm = norm(p);
            p.div(pnorm);
            t.mul(pnorm);
            w.mul(pnorm);

            num = u.transpose().mmul(t);
            var b = num.div(den);
            X.sub(t.mmul(p.transpose()));
            Y.sub(b.mmul(t).mmul(q.transpose()));


            T.addColumn(k, t);
            P.addColumn(k, p);
            U.addColumn(k, u);
            Q.addColumn(k, q);
            W.addColumn(k, w);
            B[k][k] = b;
            k++;
        }

        if(k != n) {
            T = T.subMatrix(1, n, 1, k);
            P = P.subMatrix(1, n, 1, k);
            U = U.subMatrix(1, n, 1, k);
            Q = Q.subMatrix(1, n, 1, k);
            W = W.subMatrix(1, n, 1, k);
            B = B.subMatrix(1, k, 1, k);
        }

        this.T = T;
        this.P = P;
        this.U = U;
        this.Q = Q;
        this.W = W;
        this.B = B;
    }
}

PLS.prototype.predict = function (dataset) {
    var X = Matrix(dataset);
    return (X.mmul(this.B)).add(this.B0);
};
