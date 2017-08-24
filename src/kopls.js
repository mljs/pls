import {Matrix, SingularValueDecomposition, inverse} from 'ml-matrix';
import {initializeMatrices} from './utils';

export class KOPLS {

    /**
     * Constructor for Kernel-based Orthogonal Projections to Latent Structures (K-OPLS)
     * @param {object} options
     * @param {number} [options.predictiveComponents] - Number of predictive components to use.
     * @param {number} [options.orthogonalComponents] - Number of Y-Orthogonal components.
     * @param {Kernel} [options.kernel] - Kernel object to apply.
     * @param {object} model - for load purposes.
     */
    constructor(options, model) {
        if (options === true) {
            this.X = new Matrix(model.X);
            this.Cp = new Matrix(model.Cp);
            this.Sp = new Matrix(model.Sp);
            this.Up = new Matrix(model.Up);
            this.Tp = initializeMatrices(model.Tp, false);
            this.co = initializeMatrices(model.co, false);
            this.so = model.so;
            this.to = initializeMatrices(model.to, false);
            this.toNorm = initializeMatrices(model.toNorm, false);
            this.Bt = initializeMatrices(model.Bt, false);
            this.K = initializeMatrices(model.K, true);
            this.kernel = model.kernel;
            this.orthogonalComp = model.orthogonalComp;
            this.predictiveComp = model.predictiveComp;
        } else {
            if (options.predictiveComponents === undefined) {
                throw new RangeError('no predictive components found!');
            }
            if (options.orthogonalComponents === undefined) {
                throw new RangeError('no orthogonal components found!');
            }
            if (options.kernel === undefined) {
                throw new RangeError('no kernel found!');
            }

            this.orthogonalComp = options.orthogonalComponents;
            this.predictiveComp = options.predictiveComponents;
            this.kernel = options.kernel;
        }
    }

    /**
     * Train the decision tree with the given training set and labels.
     * @param {Matrix|Array} trainingSet
     * @param {Matrix|Array} trainingValues
     */
    train(trainingSet, trainingValues) {
        trainingSet = Matrix.checkMatrix(trainingSet);
        trainingValues = Matrix.checkMatrix(trainingValues);

        // to save and compute kernel with the prediction dataset.
        this.X = trainingSet.clone();

        var KX = this.kernel.compute(trainingSet);

        var I = Matrix.eye(KX.rows, KX.rows, 1);
        var Kmc = KX.clone();
        KX = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KX[0][0] = Kmc;

        var result = new SingularValueDecomposition(trainingValues.transpose().mmul(KX[0][0]).mmul(trainingValues));
        var Cp = result.leftSingularVectors;
        var Sp = result.diagonalMatrix;

        Cp = Cp.subMatrix(0, Cp.rows - 1, 0, this.predictiveComp - 1);
        Sp = Sp.subMatrix(0, this.predictiveComp - 1, 0, this.predictiveComp - 1);

        var Up = trainingValues.mmul(Cp);

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

            to[i] = to[i].divRowVector(toNorm[i]);

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
        this.co = co;
        this.so = so;
        this.to = to;
        this.toNorm = toNorm;
        this.Bt = Bt;
        this.K = KX;
    }

    /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix|Array} toPredict
     * @return {Matrix} predictions
     */
    predict(toPredict) {

        var KteTr = this.kernel.compute(toPredict, this.X);

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
        var prediction = Tp[i].mmul(this.Bt[i]).mmul(this.Cp.transpose());

        this.predScoreMat = Tp;
        this.predYOrthVectors = to;

        return prediction;
    }

    /**
     * Get the predictive score matrix for all generations according to the number of orthogonal vectors.
     * (this can be obtained only after a prediction)
     * @return {Matrix}
     */
    getPredictiveScoreMatrix() {
        if (!this.predScoreMat) {
            throw new Error('you should run a prediction first!');
        }

        return this.predScoreMat;
    }

    /**
     * Get the predicted Y-Orthogonal vectors. (this can be obtained only after a prediction)
     * @return {Matrix}
     */
    getOrthogonalScoreVectors() {
        if (!this.predYOrthVectors) {
            throw new Error('you should run a prediction first!');
        }

        return this.predYOrthVectors;
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        return {
            name: 'K-OPLS',
            Cp: this.Cp,
            Sp: this.Sp,
            Up: this.Up,
            Tp: this.Tp,
            co: this.co,
            so: this.so,
            to: this.to,
            toNorm: this.toNorm,
            Bt: this.Bt,
            K: this.K,
            X: this.X,
            orthogonalComp: this.orthogonalComp,
            predictiveComp: this.predictiveComp
        };
    }

    /**
     * Load a K-OPLS with the given model.
     * @param {object} model
     * @param {Kernel} kernel
     * @return {KOPLS}
     */
    static load(model, kernel) {
        if (model.name !== 'K-OPLS') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        if (!kernel) {
            throw new RangeError('You must provide a kernel for the model!');
        }

        model.kernel = kernel;
        return new KOPLS(true, model);
    }
}
