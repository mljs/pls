import {Matrix, SingularValueDecomposition, inverse} from 'ml-matrix';
import Kernel from 'ml-kernel';

export class OPLS {
    constructor(options) {
        if (options.predictiveComponents === undefined) {
            throw new RangeError('no predicitive components found!');
        }
        if (options.orthogonalComponents === undefined) {
            throw new RangeError('no ortho components found!');
        }

        this.orthogonalComp = options.orthogonalComponents;
        this.predictiveComp = options.predictiveComponents;
        if (options.kernel) {
            this.kernel = new Kernel(options.kernel, options.kernelOptions);
        }

    }

    train(X, y) {
        X = Matrix.checkMatrix(X);
        y = Matrix.checkMatrix(y);

        var KX = this.kernel ? this.kernel.compute(X, X) : X;
        this.KTrain = KX;

        var I = Matrix.eye(KX.rows, KX.rows, 1);
        var Kmc = KX.clone();
        KX = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KX[0][0] = Kmc;

        var YOld = y.clone();

        var result = new SingularValueDecomposition(y.transpose().mmul(KX[0][0]).mmul(y));
        var Cp = result.leftSingularVectors;
        var Sp = result.diagonalMatrix;

        Cp = Cp.subMatrix(0, Cp.rows - 1, 0, this.predictiveComp - 1);
        Sp = Sp.subMatrix(0, this.predictiveComp - 1, 0, this.predictiveComp - 1);

        var Up = y.mmul(Cp);

        var Tp = new Array(this.orthogonalComp + 1);
        var Bt = new Array(this.orthogonalComp + 1);
        var to = new Array(this.orthogonalComp);
        var co = new Array(this.orthogonalComp);
        var so = new Array(this.orthogonalComp);
        var toNorm = new Array(this.orthogonalComp);
        for(var i = 0; i < this.orthogonalComp; ++i) {
            Tp[i] = KX[0][i].transpose().mmul(Up).mmul(Matrix.pow(Sp, -0.5));
            Bt[i] = inverse(Tp[i].transpose().mmul(Tp[i])).mmul(Tp[i].transpose()).mmul(Up);


            result = new SingularValueDecomposition(Tp[i].transpose().mmul(Matrix.sub(KX[i][i], Tp[i].mmul(Tp[i].transpose()))).mmul(Tp[i]));
            var CoTemp = result.leftSingularVectors;
            var SoTemp = result.diagonalMatrix;

            co[i] = CoTemp.subMatrix(0, CoTemp.rows - 1, 0, 0);
            so[i] = SoTemp[0][0];

            to[i] = Matrix.sub(KX[i][i], Tp[i].mmul(Tp[i].transpose())).mmul(Tp[i]).mmul(co[i]).mul(Math.pow(so[i], -0.5))
            toNorm[i] = Matrix.sqrt(to[i].transpose().mmul(to[i]));

            to[i] = to[i].divRowVector(toNorm[i]); // TODO: be careful

            var ITo = Matrix.sub(I, to[i].mmul(to[i].transpose()));

            KX[0][i + 1] = KX[0][i].mmul(ITo);
            KX[i + 1][i + 1] = ITo.mmul(KX[i][i]).mmul(ITo);
        }

        var lastTp = Tp[this.orthogonalComp] = KX[0][this.orthogonalComp].transpose().mmul(Up).mul(Math.pow(Sp, -0.5));

        Bt[this.orthogonalComp] = inverse(lastTp.transpose().mmul(lastTp)).mmul(lastTp.transpose()).mmul(Up);

        this.Cp = Cp;
        this.Sp = Sp;
        this.Up = Up;
        this.Tp = Tp;
        this.T = lastTp;
        this.co = co;
        this.so = so;
        this.to = to;
        this.toNorm = toNorm;
        this.Bt = Bt;
        this.K = KX;
    }

    predict(KteTr, Ktest) {
        var KteTrMc = KteTr;
        KteTr = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KteTr[0][0] = KteTrMc;

        var to = new Array(this.orthogonalComp);
        var Tp = new Array(this.orthogonalComp);
        var i;
        for(i = 0; i < this.orthogonalComp; ++i) {
            Tp[i] = KteTr[i][0].mmul(this.Up).mul(Math.pow(this.Sp, -0.5));

            to[i] = Matrix.sub(KteTr[i][i], Tp[i].mmul(this.Tp[i].transpose())).mmul(this.Tp[i]).mmul(this.co[i]).mul(Math.pow(this.so[i], -0.5));

            to[i] = to[i].divRowVector(this.toNorm[i]);

            KteTr[i+1][0] = Matrix.sub(KteTr[i][0], to[i].mmul(this.to[i].transpose()).mmul(this.K[0][i].transpose()));

            var p1 = Matrix.sub(KteTr[i][0], KteTr[i][i].mmul(this.to[i]).mmul(this.to[i].transpose()));
            var p2 = to[i].mmul(this.to[i].transpose()).mmul(this.K[i][i]);
            var p3 = to[i].mmul(this.to[i].transpose()).mmul(this.K[i][i]).mmul(this.to[i]).mmul(this.to[i].transpose());

            KteTr[i+1][i+1] = p1.sub(p2).add(p3);
        }

        Tp[i] = KteTr[i][0].mmul(this.Up).mul(Math.pow(this.Sp, -0.5));
        var YHat = Tp[i].mmul(this.Bt[i]).mmul(this.Cp.transpose());

        return {
            Tp: Tp,
            to: to,
            YHat: YHat
        };
    }
}

/*'use strict';

var Matrix = require('ml-matrix').Matrix;
var Utils = require('./utils');

module.exports = OPLS;

function OPLS(dataset, predictions, options = {}) {
    const {
        numberOSC = 1,
        scale = true
    } = options;
    
    var X = Matrix.checkMatrix(dataset);
    var y = Matrix.checkMatrix(predictions);
    
    if (scale) {
        X = Utils.featureNormalize(X).result;
        y = Utils.featureNormalize(y).result;
    }

    var rows = X.rows;
    var columns = X.columns;

    var sumOfSquaresX = X.clone().mul(X).sum();
    var w = X.transposeView().mmul(y);
    w.div(Utils.norm(w));

    var orthoW = new Array(numberOSC);
    var orthoT = new Array(numberOSC);
    var orthoP = new Array(numberOSC);
    for (var i = 0; i < numberOSC; i++) {
        var t = X.mmul(w);

        var numerator = X.transposeView().mmul(t);
        var denominator = t.transposeView().mmul(t)[0][0];
        var p =  numerator.div(denominator);

        numerator = w.transposeView().mmul(p)[0][0];
        denominator = w.transposeView().mmul(w)[0][0];
        var wOsc = p.sub(w.clone().mul(numerator / denominator));
        wOsc.div(Utils.norm(wOsc));

        var tOsc = X.mmul(wOsc);

        numerator = X.transposeView().mmul(tOsc);
        denominator = tOsc.transposeView().mmul(tOsc)[0][0];
        var pOsc = numerator.div(denominator);

        X.sub(tOsc.mmul(pOsc.transposeView()));
        orthoW[i] = wOsc.getColumn(0);
        orthoT[i] = tOsc.getColumn(0);
        orthoP[i] = pOsc.getColumn(0);
    }

    this.Xosc = X;

    var sumOfSquaresXosx = this.Xosc.clone().mul(this.Xosc).sum();
    this.R2X = 1 - sumOfSquaresXosx/sumOfSquaresX;

    this.W = orthoW;
    this.T = orthoT;
    this.P = orthoP;
    this.numberOSC = numberOSC;
}

OPLS.prototype.correctDataset = function (dataset) {
    var X = new Matrix(dataset);

    var sumOfSquaresX = X.clone().mul(X).sum();
    for (var i = 0; i < this.numberOSC; i++) {
        var currentW = this.W.getColumnVector(i);
        var currentP = this.P.getColumnVector(i);

        var t = X.mmul(currentW);
        X.sub(t.mmul(currentP));
    }
    var sumOfSquaresXosx = X.clone().mul(X).sum();

    var R2X = 1 - sumOfSquaresXosx / sumOfSquaresX;

    return {
        datasetOsc: X,
        R2Dataset: R2X
    };
};
*/