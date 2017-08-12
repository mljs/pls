import Matrix from 'ml-matrix';
import Kernel from 'ml-kernel';
import {OPLS} from '..';

describe('OPLS', () => {
    var Xtest = new Matrix(require('./Xtest.json'));
    var Xtrain = new Matrix(require('./Xtrain.json'));
    var Ytest = new Matrix(require('./Ytest.json'));
    var Ytrain = new Matrix(require('./Ytrain.json'));

    test('main test', () => {
        var cls = new OPLS({
            orthogonalComponents: 10,
            predictiveComponents: 1,
            kernel: 'gaussian',
            kernelOptions: {
                sigma: 25
            }
        });
        cls.train(Xtrain, Ytrain);
        var currKernel = new Kernel('gaussian', {
            sigma: 25
        });
        var KteTr = currKernel.compute(Xtest, Xtrain);
        var KteTe = currKernel.compute(Xtest, Xtest);

        var output = cls.predict(KteTr, KteTe).YHat;

        for(var i = 0; i < output.rows; ++i) {
            for(var j = 0; j < output.columns; ++j) {
                expect(output[i][j]).toBeCloseTo(Ytest[i][j], 1);
            }
        }
    });
});
