import {Matrix, SingularValueDecomposition, inverse} from 'ml-matrix';
import Kernel from 'ml-kernel';

export class KOPLS {
    constructor(options) {
        if (options.predictiveComponents === undefined) {
            throw new RangeError('no predicitive components found!');
        }
        if (options.orthogonalComponents === undefined) {
            throw new RangeError('no orthogonal components found!');
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

        this.X = X.clone();
        var KX = this.kernel ? this.kernel.compute(X, X) : X;

        var I = Matrix.eye(KX.rows, KX.rows, 1);
        var Kmc = KX.clone();
        KX = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KX[0][0] = Kmc;

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

        var SpPow = Matrix.pow(Sp, -0.5);
        for (var i = 0; i < this.orthogonalComp; ++i) {
            Tp[i] = KX[0][i].transpose().mmul(Up).mmul(SpPow);

            var TpiPrime = Tp[i].transpose();
            Bt[i] = inverse(TpiPrime.mmul(Tp[i])).mmul(TpiPrime).mmul(Up);

            result = new SingularValueDecomposition(TpiPrime.mmul(Matrix.sub(KX[i][i], Tp[i].mmul(TpiPrime))).mmul(Tp[i]));
            var CoTemp = result.leftSingularVectors;
            var SoTemp = result.diagonalMatrix;

            co[i] = CoTemp.subMatrix(0, CoTemp.rows - 1, 0, 0);
            so[i] = SoTemp[0][0];

            to[i] = Matrix.sub(KX[i][i], Tp[i].mmul(TpiPrime)).mmul(Tp[i]).mmul(co[i]).mul(Math.pow(so[i], -0.5));

            var toiPrime = to[i].transpose();
            toNorm[i] = Matrix.sqrt(toiPrime.mmul(to[i]));

            to[i] = to[i].divRowVector(toNorm[i]); // TODO: be careful

            var ITo = Matrix.sub(I, to[i].mmul(to[i].transpose()));

            KX[0][i + 1] = KX[0][i].mmul(ITo);
            KX[i + 1][i + 1] = ITo.mmul(KX[i][i]).mmul(ITo);
        }

        var lastTp = Tp[this.orthogonalComp] = KX[0][this.orthogonalComp].transpose().mmul(Up).mul(Math.pow(Sp, -0.5));

        var lastTpPrime = lastTp.transpose();
        Bt[this.orthogonalComp] = inverse(lastTpPrime.mmul(lastTp)).mmul(lastTpPrime).mmul(Up);

        this.Cp = Cp;
        this.Sp = Sp;
        this.Up = Up;
        this.Tp = Tp;
        // this.T = lastTp;
        this.co = co;
        this.so = so;
        this.to = to;
        this.toNorm = toNorm;
        this.Bt = Bt;
        this.K = KX;
    }

    predict(X) {

        var KteTr = this.kernel ? this.kernel.compute(X, this.X) : X;

        var KteTrMc = KteTr;
        KteTr = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KteTr[0][0] = KteTrMc;

        var to = new Array(this.orthogonalComp);
        var Tp = new Array(this.orthogonalComp);
        var SpPow = Math.pow(this.Sp, -0.5);

        var i;
        for (i = 0; i < this.orthogonalComp; ++i) {
            Tp[i] = KteTr[i][0].mmul(this.Up).mul(SpPow);

            to[i] = Matrix.sub(KteTr[i][i], Tp[i].mmul(this.Tp[i].transpose())).mmul(this.Tp[i]).mmul(this.co[i]).mul(Math.pow(this.so[i], -0.5));

            to[i] = to[i].divRowVector(this.toNorm[i]);

            var toiPrime = this.to[i].transpose();
            KteTr[i + 1][0] = Matrix.sub(KteTr[i][0], to[i].mmul(toiPrime).mmul(this.K[0][i].transpose()));

            var p1 = Matrix.sub(KteTr[i][0], KteTr[i][i].mmul(this.to[i]).mmul(toiPrime));
            var p2 = to[i].mmul(toiPrime).mmul(this.K[i][i]);
            var p3 = p2.mmul(this.to[i]).mmul(toiPrime);

            KteTr[i + 1][i + 1] = p1.sub(p2).add(p3);
        }

        Tp[i] = KteTr[i][0].mmul(this.Up).mul(SpPow);
        var YHat = Tp[i].mmul(this.Bt[i]).mmul(this.Cp.transpose());

        return {
            Tp: Tp,
            to: to,
            YHat: YHat
        };
    }
}
