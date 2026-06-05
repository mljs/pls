var Suite = require('benchmark').Suite;
var PLS = require('..');
var Matrix = require('ml-matrix');

var dataset = new Matrix([[7, 7, 13, 7],
    [4, 3, 14, 7],
    [10, 5, 12, 5],
    [16, 7, 11, 3],
    [13, 3, 10, 3]]);

var suite = new Suite();

suite
    .add('v1', function () {
        maxSumColIndex(dataset);
    })
    .add('v2', function () {
        maxSumColIndex2(dataset);
    })
    .add('v3', function () {
        maxSumColIndex3(dataset);
    })
    .on('cycle', function (event) {
        console.log(String(event.target));
    })
    .run();


function getColSum(matrix, column) {
    var sum = 0;
    for (var i = 0; i < matrix.rows; i++) {
        sum += matrix[i][column];
    }
    return sum;
}

function maxSumColIndex3(data) {
    var maxIndex = 0;
    var maxSum = -Infinity;
    for(var column = 0; column < data.columns; ++column) {
        var currentSum = 0
        for (var i = 0; i < data.rows; i++) {
            currentSum += data[i][column];
        }
        if(currentSum > maxSum) {
            maxSum = currentSum;
            maxIndex = i;
        }
    }
    return maxIndex;
}

function maxSumColIndex2(data) {
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

/**
 * Function that returns the index where the sum of each
 * column vector is maximum.
 * @param {Matrix} X
 * @returns {number} index of the maximum
 */
function maxSumColIndex(X) {
    var maxIndex = 0;
    var maxSum = -Infinity;
    for(var i = 0; i < X.columns; ++i) {
        var currentSum = X.getColumnVector(i).sum();
        if(currentSum > maxSum) {
            maxSum = currentSum;
            maxIndex = i;
        }
    }
    return maxIndex;
}