import Matrix from 'ml-matrix';
import {KOPLS} from '..';

describe('OPLS', () => {
    var Xtest = new Matrix(require('./Xtest.json'));
    var Xtrain = new Matrix(require('./Xtrain.json'));
    var Ytest = new Matrix(require('./Ytest.json'));
    var Ytrain = new Matrix(require('./Ytrain.json'));

    test('main test', () => {
        var cls = new KOPLS({
            orthogonalComponents: 10,
            predictiveComponents: 1, // TODO: test with more predictive components
            kernel: 'gaussian',
            kernelOptions: {
                sigma: 25
            }
        });
        cls.train(Xtrain, Ytrain);

        var output = cls.predict(Xtest).YHat;

        for (var i = 0; i < output.rows; ++i) {
            for (var j = 0; j < output.columns; ++j) {
                expect(output[i][j]).toBeCloseTo(Ytest[i][j], 3);
            }
        }
    });
});
