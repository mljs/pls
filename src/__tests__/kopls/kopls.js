import Matrix from 'ml-matrix';
import Kernel from 'ml-kernel';
import {KOPLS} from '../../index';

describe('K-OPLS', () => {
    var Xtest = new Matrix(require('./Xtest.json'));
    var Xtrain = new Matrix(require('./Xtrain.json'));
    var Ytest = new Matrix(require('./Ytest.json'));
    var Ytrain = new Matrix(require('./Ytrain.json'));
    var Tp = new Matrix(require('./tp.json'));
    var to = new Matrix(require('./to.json'));

    var kernel = new Kernel('gaussian', {
        sigma: 25
    });

    var cls = new KOPLS({
        orthogonalComponents: 10,
        predictiveComponents: 1,
        kernel: kernel
    });
    cls.train(Xtrain, Ytrain);

    test('K-OPLS test with main features', () => {
        expect(() => cls.getOrthogonalScoreVectors()).toThrowError(Error);
        expect(() => cls.getPredictiveScoreMatrix()).toThrowError(Error);

        var output = cls.predict(Xtest);

        var testTp = cls.getPredictiveScoreMatrix();
        for (var i = 0; i < testTp.length; ++i) {
            for (var j = 0; j < testTp[i].length; ++j) {
                expect(testTp[i][j][0]).toBeCloseTo(Tp[i][j], 2);
            }
        }

        var testTo = cls.getOrthogonalScoreVectors();
        for (i = 0; i < testTo.length; ++i) {
            for (j = 0; j < testTo[i].length; ++j) {
                expect(testTo[i][j][0]).toBeCloseTo(to[i][j], 2);
            }
        }

        for (i = 0; i < output.rows; ++i) {
            for (j = 0; j < output.columns; ++j) {
                expect(output[i][j]).toBeCloseTo(Ytest[i][j], 3);
            }
        }
    });

    test('Load and save', () => {
        var model = KOPLS.load(JSON.parse(JSON.stringify(cls)), kernel);
        var output = model.predict(Xtest);

        for (var i = 0; i < output.rows; ++i) {
            for (var j = 0; j < output.columns; ++j) {
                expect(output[i][j]).toBeCloseTo(Ytest[i][j], 3);
            }
        }
    });
});
