import {Matrix, SingularValueDecomposition, inverse} from 'ml-matrix';
import {initializeMatrices} from './utils';

/**
 * @class KOPLS
 */
export class KOPLS {

    /**
     * Constructor for Kernel-based Orthogonal Projections to Latent Structures (K-OPLS)
     * @param {object} options
     * @param {number} [options.predictiveComponents] - Number of predictive components to use.
     * @param {number} [options.orthogonalComponents] - Number of Y-Orthogonal components.
     * @param {Kernel} [options.kernel] - Kernel object to apply, see [ml-kernel](https://github.com/mljs/kernel).
     * @param {object} model - for load purposes.
     */
    constructor(options, model) {
        if (options === true) {
            this.trainingSet = new Matrix(model.trainingSet);
            this.YLoadingMat = new Matrix(model.YLoadingMat);
            this.SigmaPow = new Matrix(model.SigmaPow);
            this.YScoreMat = new Matrix(model.YScoreMat);
            this.predScoreMat = initializeMatrices(model.predScoreMat, false);
            this.YOrthLoadingVec = initializeMatrices(model.YOrthLoadingVec, false);
            this.YOrthEigen = model.YOrthEigen;
            this.YOrthScoreMat = initializeMatrices(model.YOrthScoreMat, false);
            this.toNorm = initializeMatrices(model.toNorm, false);
            this.TURegressionCoeff = initializeMatrices(model.TURegressionCoeff, false);
            this.kernelX = initializeMatrices(model.kernelX, true);
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
     * Train the K-OPLS model with the given training set and labels.
     * @param {Matrix|Array} trainingSet
     * @param {Matrix|Array} trainingValues
     */
    train(trainingSet, trainingValues) {
        trainingSet = Matrix.checkMatrix(trainingSet);
        trainingValues = Matrix.checkMatrix(trainingValues);

        // to save and compute kernel with the prediction dataset.
        this.trainingSet = trainingSet.clone();

        var kernelX = this.kernel.compute(trainingSet);

        var Identity = Matrix.eye(kernelX.rows, kernelX.rows, 1);
        var temp = kernelX;
        kernelX = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        kernelX[0][0] = temp;

        var result = new SingularValueDecomposition(trainingValues.transpose().mmul(kernelX[0][0]).mmul(trainingValues), {
            computeLeftSingularVectors: true,
            computeRightSingularVectors: false
        });
        var YLoadingMat = result.leftSingularVectors;
        var Sigma = result.diagonalMatrix;

        YLoadingMat = YLoadingMat.subMatrix(0, YLoadingMat.rows - 1, 0, this.predictiveComp - 1);
        Sigma = Sigma.subMatrix(0, this.predictiveComp - 1, 0, this.predictiveComp - 1);

        var YScoreMat = trainingValues.mmul(YLoadingMat);

        var predScoreMat = new Array(this.orthogonalComp + 1);
        var TURegressionCoeff = new Array(this.orthogonalComp + 1);
        var YOrthScoreMat = new Array(this.orthogonalComp);
        var YOrthLoadingVec = new Array(this.orthogonalComp);
        var YOrthEigen = new Array(this.orthogonalComp);
        var YOrthScoreNorm = new Array(this.orthogonalComp);

        var SigmaPow = Matrix.pow(Sigma, -0.5);
        // to avoid errors, check infinity
        SigmaPow.apply(function (i, j) {
            if (this[i][j] === Infinity) {
                this[i][j] = 0.0;
            }
        });

        for (var i = 0; i < this.orthogonalComp; ++i) {
            predScoreMat[i] = kernelX[0][i].transpose().mmul(YScoreMat).mmul(SigmaPow);

            var TpiPrime = predScoreMat[i].transpose();
            TURegressionCoeff[i] = inverse(TpiPrime.mmul(predScoreMat[i])).mmul(TpiPrime).mmul(YScoreMat);

            result = new SingularValueDecomposition(TpiPrime.mmul(Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime))).mmul(predScoreMat[i]), {
                computeLeftSingularVectors: true,
                computeRightSingularVectors: false
            });
            var CoTemp = result.leftSingularVectors;
            var SoTemp = result.diagonalMatrix;

            YOrthLoadingVec[i] = CoTemp.subMatrix(0, CoTemp.rows - 1, 0, 0);
            YOrthEigen[i] = SoTemp[0][0];

            YOrthScoreMat[i] = Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime)).mmul(predScoreMat[i]).mmul(YOrthLoadingVec[i]).mul(Math.pow(YOrthEigen[i], -0.5));

            var toiPrime = YOrthScoreMat[i].transpose();
            YOrthScoreNorm[i] = Matrix.sqrt(toiPrime.mmul(YOrthScoreMat[i]));

            YOrthScoreMat[i] = YOrthScoreMat[i].divRowVector(YOrthScoreNorm[i]);

            var ITo = Matrix.sub(Identity, YOrthScoreMat[i].mmul(YOrthScoreMat[i].transpose()));

            kernelX[0][i + 1] = kernelX[0][i].mmul(ITo);
            kernelX[i + 1][i + 1] = ITo.mmul(kernelX[i][i]).mmul(ITo);
        }

        var lastScoreMat = predScoreMat[this.orthogonalComp] = kernelX[0][this.orthogonalComp].transpose().mmul(YScoreMat).mmul(SigmaPow);

        var lastTpPrime = lastScoreMat.transpose();
        TURegressionCoeff[this.orthogonalComp] = inverse(lastTpPrime.mmul(lastScoreMat)).mmul(lastTpPrime).mmul(YScoreMat);

        this.YLoadingMat = YLoadingMat;
        this.SigmaPow = SigmaPow;
        this.YScoreMat = YScoreMat;
        this.predScoreMat = predScoreMat;
        this.YOrthLoadingVec = YOrthLoadingVec;
        this.YOrthEigen = YOrthEigen;
        this.YOrthScoreMat = YOrthScoreMat;
        this.toNorm = YOrthScoreNorm;
        this.TURegressionCoeff = TURegressionCoeff;
        this.kernelX = kernelX;
    }

    /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix|Array} toPredict
     * @return {{y: Matrix, predScoreMat: Array<Matrix>, predYOrthVectors: Array<Matrix>}} predictions
     */
    predict(toPredict) {

        var KTestTrain = this.kernel.compute(toPredict, this.trainingSet);

        var temp = KTestTrain;
        KTestTrain = new Matrix(this.orthogonalComp + 1, this.orthogonalComp + 1);
        KTestTrain[0][0] = temp;

        var YOrthScoreVector = new Array(this.orthogonalComp);
        var predScoreMat = new Array(this.orthogonalComp);

        var i;
        for (i = 0; i < this.orthogonalComp; ++i) {
            predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);

            YOrthScoreVector[i] = Matrix.sub(KTestTrain[i][i], predScoreMat[i].mmul(this.predScoreMat[i].transpose())).mmul(this.predScoreMat[i]).mmul(this.YOrthLoadingVec[i]).mul(Math.pow(this.YOrthEigen[i], -0.5));

            YOrthScoreVector[i] = YOrthScoreVector[i].divRowVector(this.toNorm[i]);

            var scoreMatPrime = this.YOrthScoreMat[i].transpose();
            KTestTrain[i + 1][0] = Matrix.sub(KTestTrain[i][0], YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[0][i].transpose()));

            var p1 = Matrix.sub(KTestTrain[i][0], KTestTrain[i][i].mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime));
            var p2 = YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[i][i]);
            var p3 = p2.mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime);

            KTestTrain[i + 1][i + 1] = p1.sub(p2).add(p3);
        }

        predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);
        var prediction = predScoreMat[i].mmul(this.TURegressionCoeff[i]).mmul(this.YLoadingMat.transpose());

        return {
            prediction: prediction,
            predScoreMat: predScoreMat,
            predYOrthVectors: YOrthScoreVector
        };
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        return {
            name: 'K-OPLS',
            YLoadingMat: this.YLoadingMat,
            SigmaPow: this.SigmaPow,
            YScoreMat: this.YScoreMat,
            predScoreMat: this.predScoreMat,
            YOrthLoadingVec: this.YOrthLoadingVec,
            YOrthEigen: this.YOrthEigen,
            YOrthScoreMat: this.YOrthScoreMat,
            toNorm: this.toNorm,
            TURegressionCoeff: this.TURegressionCoeff,
            kernelX: this.kernelX,
            trainingSet: this.trainingSet,
            orthogonalComp: this.orthogonalComp,
            predictiveComp: this.predictiveComp
        };
    }

    /**
     * Load a K-OPLS with the given model.
     * @param {object} model
     * @param {Kernel} kernel - kernel used on the model, see [ml-kernel](https://github.com/mljs/kernel).
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
