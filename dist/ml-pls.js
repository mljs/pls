/**
 * ml-pls - Partial least squares library
 * @version v2.0.0
 * @link https://github.com/mljs/pls
 * @license MIT
 */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = global || self, factory(global.mlPls = {}));
}(this, (function (exports) { 'use strict';

  const toString = Object.prototype.toString;

  function isAnyArray(object) {
    return toString.call(object).endsWith('Array]');
  }

  var src = isAnyArray;

  /**
   * Computes the maximum of the given values
   * @param {Array<number>} input
   * @return {number}
   */

  function max(input) {
    if (!src(input)) {
      throw new TypeError('input must be an array');
    }

    if (input.length === 0) {
      throw new TypeError('input must not be empty');
    }

    var maxValue = input[0];

    for (var i = 1; i < input.length; i++) {
      if (input[i] > maxValue) maxValue = input[i];
    }

    return maxValue;
  }

  /**
   * Computes the minimum of the given values
   * @param {Array<number>} input
   * @return {number}
   */

  function min(input) {
    if (!src(input)) {
      throw new TypeError('input must be an array');
    }

    if (input.length === 0) {
      throw new TypeError('input must not be empty');
    }

    var minValue = input[0];

    for (var i = 1; i < input.length; i++) {
      if (input[i] < minValue) minValue = input[i];
    }

    return minValue;
  }

  function rescale(input) {
    var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

    if (!src(input)) {
      throw new TypeError('input must be an array');
    } else if (input.length === 0) {
      throw new TypeError('input must not be empty');
    }

    var output;

    if (options.output !== undefined) {
      if (!src(options.output)) {
        throw new TypeError('output option must be an array if specified');
      }

      output = options.output;
    } else {
      output = new Array(input.length);
    }

    var currentMin = min(input);
    var currentMax = max(input);

    if (currentMin === currentMax) {
      throw new RangeError('minimum and maximum input values are equal. Cannot rescale a constant array');
    }

    var _options$min = options.min,
        minValue = _options$min === void 0 ? options.autoMinMax ? currentMin : 0 : _options$min,
        _options$max = options.max,
        maxValue = _options$max === void 0 ? options.autoMinMax ? currentMax : 1 : _options$max;

    if (minValue >= maxValue) {
      throw new RangeError('min option must be smaller than max option');
    }

    var factor = (maxValue - minValue) / (currentMax - currentMin);

    for (var i = 0; i < input.length; i++) {
      output[i] = (input[i] - currentMin) * factor + minValue;
    }

    return output;
  }

  /**
   * @private
   * Check that a row index is not out of bounds
   * @param {Matrix} matrix
   * @param {number} index
   * @param {boolean} [outer]
   */
  function checkRowIndex(matrix, index, outer) {
    let max = outer ? matrix.rows : matrix.rows - 1;

    if (index < 0 || index > max) {
      throw new RangeError('Row index out of range');
    }
  }
  /**
   * @private
   * Check that a column index is not out of bounds
   * @param {Matrix} matrix
   * @param {number} index
   * @param {boolean} [outer]
   */

  function checkColumnIndex(matrix, index, outer) {
    let max = outer ? matrix.columns : matrix.columns - 1;

    if (index < 0 || index > max) {
      throw new RangeError('Column index out of range');
    }
  }
  /**
   * @private
   * Check that the provided vector is an array with the right length
   * @param {Matrix} matrix
   * @param {Array|Matrix} vector
   * @return {Array}
   * @throws {RangeError}
   */

  function checkRowVector(matrix, vector) {
    if (vector.to1DArray) {
      vector = vector.to1DArray();
    }

    if (vector.length !== matrix.columns) {
      throw new RangeError('vector size must be the same as the number of columns');
    }

    return vector;
  }
  /**
   * @private
   * Check that the provided vector is an array with the right length
   * @param {Matrix} matrix
   * @param {Array|Matrix} vector
   * @return {Array}
   * @throws {RangeError}
   */

  function checkColumnVector(matrix, vector) {
    if (vector.to1DArray) {
      vector = vector.to1DArray();
    }

    if (vector.length !== matrix.rows) {
      throw new RangeError('vector size must be the same as the number of rows');
    }

    return vector;
  }
  function checkIndices(matrix, rowIndices, columnIndices) {
    return {
      row: checkRowIndices(matrix, rowIndices),
      column: checkColumnIndices(matrix, columnIndices)
    };
  }
  function checkRowIndices(matrix, rowIndices) {
    if (typeof rowIndices !== 'object') {
      throw new TypeError('unexpected type for row indices');
    }

    let rowOut = rowIndices.some(r => {
      return r < 0 || r >= matrix.rows;
    });

    if (rowOut) {
      throw new RangeError('row indices are out of range');
    }

    if (!Array.isArray(rowIndices)) rowIndices = Array.from(rowIndices);
    return rowIndices;
  }
  function checkColumnIndices(matrix, columnIndices) {
    if (typeof columnIndices !== 'object') {
      throw new TypeError('unexpected type for column indices');
    }

    let columnOut = columnIndices.some(c => {
      return c < 0 || c >= matrix.columns;
    });

    if (columnOut) {
      throw new RangeError('column indices are out of range');
    }

    if (!Array.isArray(columnIndices)) columnIndices = Array.from(columnIndices);
    return columnIndices;
  }
  function checkRange(matrix, startRow, endRow, startColumn, endColumn) {
    if (arguments.length !== 5) {
      throw new RangeError('expected 4 arguments');
    }

    checkNumber('startRow', startRow);
    checkNumber('endRow', endRow);
    checkNumber('startColumn', startColumn);
    checkNumber('endColumn', endColumn);

    if (startRow > endRow || startColumn > endColumn || startRow < 0 || startRow >= matrix.rows || endRow < 0 || endRow >= matrix.rows || startColumn < 0 || startColumn >= matrix.columns || endColumn < 0 || endColumn >= matrix.columns) {
      throw new RangeError('Submatrix indices are out of range');
    }
  }
  function newArray(length) {
    let value = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
    let array = [];

    for (let i = 0; i < length; i++) {
      array.push(value);
    }

    return array;
  }

  function checkNumber(name, value) {
    if (typeof value !== 'number') {
      throw new TypeError("".concat(name, " must be a number"));
    }
  }

  function sumByRow(matrix) {
    let sum = newArray(matrix.rows);

    for (let i = 0; i < matrix.rows; ++i) {
      for (let j = 0; j < matrix.columns; ++j) {
        sum[i] += matrix.get(i, j);
      }
    }

    return sum;
  }
  function sumByColumn(matrix) {
    let sum = newArray(matrix.columns);

    for (let i = 0; i < matrix.rows; ++i) {
      for (let j = 0; j < matrix.columns; ++j) {
        sum[j] += matrix.get(i, j);
      }
    }

    return sum;
  }
  function sumAll(matrix) {
    let v = 0;

    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        v += matrix.get(i, j);
      }
    }

    return v;
  }
  function productByRow(matrix) {
    let sum = newArray(matrix.rows, 1);

    for (let i = 0; i < matrix.rows; ++i) {
      for (let j = 0; j < matrix.columns; ++j) {
        sum[i] *= matrix.get(i, j);
      }
    }

    return sum;
  }
  function productByColumn(matrix) {
    let sum = newArray(matrix.columns, 1);

    for (let i = 0; i < matrix.rows; ++i) {
      for (let j = 0; j < matrix.columns; ++j) {
        sum[j] *= matrix.get(i, j);
      }
    }

    return sum;
  }
  function productAll(matrix) {
    let v = 1;

    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        v *= matrix.get(i, j);
      }
    }

    return v;
  }
  function varianceByRow(matrix, unbiased, mean) {
    const rows = matrix.rows;
    const cols = matrix.columns;
    const variance = [];

    for (let i = 0; i < rows; i++) {
      let sum1 = 0;
      let sum2 = 0;
      let x = 0;

      for (let j = 0; j < cols; j++) {
        x = matrix.get(i, j) - mean[i];
        sum1 += x;
        sum2 += x * x;
      }

      if (unbiased) {
        variance.push((sum2 - sum1 * sum1 / cols) / (cols - 1));
      } else {
        variance.push((sum2 - sum1 * sum1 / cols) / cols);
      }
    }

    return variance;
  }
  function varianceByColumn(matrix, unbiased, mean) {
    const rows = matrix.rows;
    const cols = matrix.columns;
    const variance = [];

    for (let j = 0; j < cols; j++) {
      let sum1 = 0;
      let sum2 = 0;
      let x = 0;

      for (let i = 0; i < rows; i++) {
        x = matrix.get(i, j) - mean[j];
        sum1 += x;
        sum2 += x * x;
      }

      if (unbiased) {
        variance.push((sum2 - sum1 * sum1 / rows) / (rows - 1));
      } else {
        variance.push((sum2 - sum1 * sum1 / rows) / rows);
      }
    }

    return variance;
  }
  function varianceAll(matrix, unbiased, mean) {
    const rows = matrix.rows;
    const cols = matrix.columns;
    const size = rows * cols;
    let sum1 = 0;
    let sum2 = 0;
    let x = 0;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        x = matrix.get(i, j) - mean;
        sum1 += x;
        sum2 += x * x;
      }
    }

    if (unbiased) {
      return (sum2 - sum1 * sum1 / size) / (size - 1);
    } else {
      return (sum2 - sum1 * sum1 / size) / size;
    }
  }
  function centerByRow(matrix, mean) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) - mean[i]);
      }
    }
  }
  function centerByColumn(matrix, mean) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) - mean[j]);
      }
    }
  }
  function centerAll(matrix, mean) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) - mean);
      }
    }
  }
  function getScaleByRow(matrix) {
    const scale = [];

    for (let i = 0; i < matrix.rows; i++) {
      let sum = 0;

      for (let j = 0; j < matrix.columns; j++) {
        sum += Math.pow(matrix.get(i, j), 2) / (matrix.columns - 1);
      }

      scale.push(Math.sqrt(sum));
    }

    return scale;
  }
  function scaleByRow(matrix, scale) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) / scale[i]);
      }
    }
  }
  function getScaleByColumn(matrix) {
    const scale = [];

    for (let j = 0; j < matrix.columns; j++) {
      let sum = 0;

      for (let i = 0; i < matrix.rows; i++) {
        sum += Math.pow(matrix.get(i, j), 2) / (matrix.rows - 1);
      }

      scale.push(Math.sqrt(sum));
    }

    return scale;
  }
  function scaleByColumn(matrix, scale) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) / scale[j]);
      }
    }
  }
  function getScaleAll(matrix) {
    const divider = matrix.size - 1;
    let sum = 0;

    for (let j = 0; j < matrix.columns; j++) {
      for (let i = 0; i < matrix.rows; i++) {
        sum += Math.pow(matrix.get(i, j), 2) / divider;
      }
    }

    return Math.sqrt(sum);
  }
  function scaleAll(matrix, scale) {
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        matrix.set(i, j, matrix.get(i, j) / scale);
      }
    }
  }

  function inspectMatrix() {
    const indent = ' '.repeat(2);
    const indentData = ' '.repeat(4);
    return "".concat(this.constructor.name, " {\n").concat(indent, "[\n").concat(indentData).concat(inspectData(this, indentData), "\n").concat(indent, "]\n").concat(indent, "rows: ").concat(this.rows, "\n").concat(indent, "columns: ").concat(this.columns, "\n}");
  }
  const maxRows = 15;
  const maxColumns = 10;
  const maxNumSize = 8;

  function inspectData(matrix, indent) {
    const {
      rows,
      columns
    } = matrix;
    const maxI = Math.min(rows, maxRows);
    const maxJ = Math.min(columns, maxColumns);
    const result = [];

    for (let i = 0; i < maxI; i++) {
      let line = [];

      for (let j = 0; j < maxJ; j++) {
        line.push(formatNumber(matrix.get(i, j)));
      }

      result.push("".concat(line.join(' ')));
    }

    if (maxJ !== columns) {
      result[result.length - 1] += " ... ".concat(columns - maxColumns, " more columns");
    }

    if (maxI !== rows) {
      result.push("... ".concat(rows - maxRows, " more rows"));
    }

    return result.join("\n".concat(indent));
  }

  function formatNumber(num) {
    const numStr = String(num);

    if (numStr.length <= maxNumSize) {
      return numStr.padEnd(maxNumSize, ' ');
    }

    const precise = num.toPrecision(maxNumSize - 2);

    if (precise.length <= maxNumSize) {
      return precise;
    }

    const exponential = num.toExponential(maxNumSize - 2);
    const eIndex = exponential.indexOf('e');
    const e = exponential.substring(eIndex);
    return exponential.substring(0, maxNumSize - e.length) + e;
  }

  function installMathOperations(AbstractMatrix, Matrix) {
    AbstractMatrix.prototype.add = function add(value) {
      if (typeof value === 'number') return this.addS(value);
      return this.addM(value);
    };

    AbstractMatrix.prototype.addS = function addS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) + value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.addM = function addM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) + matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.add = function add(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.add(value);
    };

    AbstractMatrix.prototype.sub = function sub(value) {
      if (typeof value === 'number') return this.subS(value);
      return this.subM(value);
    };

    AbstractMatrix.prototype.subS = function subS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) - value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.subM = function subM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) - matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.sub = function sub(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.sub(value);
    };

    AbstractMatrix.prototype.subtract = AbstractMatrix.prototype.sub;
    AbstractMatrix.prototype.subtractS = AbstractMatrix.prototype.subS;
    AbstractMatrix.prototype.subtractM = AbstractMatrix.prototype.subM;
    AbstractMatrix.subtract = AbstractMatrix.sub;

    AbstractMatrix.prototype.mul = function mul(value) {
      if (typeof value === 'number') return this.mulS(value);
      return this.mulM(value);
    };

    AbstractMatrix.prototype.mulS = function mulS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) * value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.mulM = function mulM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) * matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.mul = function mul(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.mul(value);
    };

    AbstractMatrix.prototype.multiply = AbstractMatrix.prototype.mul;
    AbstractMatrix.prototype.multiplyS = AbstractMatrix.prototype.mulS;
    AbstractMatrix.prototype.multiplyM = AbstractMatrix.prototype.mulM;
    AbstractMatrix.multiply = AbstractMatrix.mul;

    AbstractMatrix.prototype.div = function div(value) {
      if (typeof value === 'number') return this.divS(value);
      return this.divM(value);
    };

    AbstractMatrix.prototype.divS = function divS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) / value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.divM = function divM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) / matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.div = function div(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.div(value);
    };

    AbstractMatrix.prototype.divide = AbstractMatrix.prototype.div;
    AbstractMatrix.prototype.divideS = AbstractMatrix.prototype.divS;
    AbstractMatrix.prototype.divideM = AbstractMatrix.prototype.divM;
    AbstractMatrix.divide = AbstractMatrix.div;

    AbstractMatrix.prototype.mod = function mod(value) {
      if (typeof value === 'number') return this.modS(value);
      return this.modM(value);
    };

    AbstractMatrix.prototype.modS = function modS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) % value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.modM = function modM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) % matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.mod = function mod(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.mod(value);
    };

    AbstractMatrix.prototype.modulus = AbstractMatrix.prototype.mod;
    AbstractMatrix.prototype.modulusS = AbstractMatrix.prototype.modS;
    AbstractMatrix.prototype.modulusM = AbstractMatrix.prototype.modM;
    AbstractMatrix.modulus = AbstractMatrix.mod;

    AbstractMatrix.prototype.and = function and(value) {
      if (typeof value === 'number') return this.andS(value);
      return this.andM(value);
    };

    AbstractMatrix.prototype.andS = function andS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) & value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.andM = function andM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) & matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.and = function and(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.and(value);
    };

    AbstractMatrix.prototype.or = function or(value) {
      if (typeof value === 'number') return this.orS(value);
      return this.orM(value);
    };

    AbstractMatrix.prototype.orS = function orS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) | value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.orM = function orM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) | matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.or = function or(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.or(value);
    };

    AbstractMatrix.prototype.xor = function xor(value) {
      if (typeof value === 'number') return this.xorS(value);
      return this.xorM(value);
    };

    AbstractMatrix.prototype.xorS = function xorS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) ^ value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.xorM = function xorM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) ^ matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.xor = function xor(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.xor(value);
    };

    AbstractMatrix.prototype.leftShift = function leftShift(value) {
      if (typeof value === 'number') return this.leftShiftS(value);
      return this.leftShiftM(value);
    };

    AbstractMatrix.prototype.leftShiftS = function leftShiftS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) << value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.leftShiftM = function leftShiftM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) << matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.leftShift = function leftShift(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.leftShift(value);
    };

    AbstractMatrix.prototype.signPropagatingRightShift = function signPropagatingRightShift(value) {
      if (typeof value === 'number') return this.signPropagatingRightShiftS(value);
      return this.signPropagatingRightShiftM(value);
    };

    AbstractMatrix.prototype.signPropagatingRightShiftS = function signPropagatingRightShiftS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) >> value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.signPropagatingRightShiftM = function signPropagatingRightShiftM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) >> matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.signPropagatingRightShift = function signPropagatingRightShift(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.signPropagatingRightShift(value);
    };

    AbstractMatrix.prototype.rightShift = function rightShift(value) {
      if (typeof value === 'number') return this.rightShiftS(value);
      return this.rightShiftM(value);
    };

    AbstractMatrix.prototype.rightShiftS = function rightShiftS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) >>> value);
        }
      }

      return this;
    };

    AbstractMatrix.prototype.rightShiftM = function rightShiftM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) >>> matrix.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.rightShift = function rightShift(matrix, value) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.rightShift(value);
    };

    AbstractMatrix.prototype.zeroFillRightShift = AbstractMatrix.prototype.rightShift;
    AbstractMatrix.prototype.zeroFillRightShiftS = AbstractMatrix.prototype.rightShiftS;
    AbstractMatrix.prototype.zeroFillRightShiftM = AbstractMatrix.prototype.rightShiftM;
    AbstractMatrix.zeroFillRightShift = AbstractMatrix.rightShift;

    AbstractMatrix.prototype.not = function not() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, ~this.get(i, j));
        }
      }

      return this;
    };

    AbstractMatrix.not = function not(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.not();
    };

    AbstractMatrix.prototype.abs = function abs() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.abs(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.abs = function abs(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.abs();
    };

    AbstractMatrix.prototype.acos = function acos() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.acos(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.acos = function acos(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.acos();
    };

    AbstractMatrix.prototype.acosh = function acosh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.acosh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.acosh = function acosh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.acosh();
    };

    AbstractMatrix.prototype.asin = function asin() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.asin(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.asin = function asin(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.asin();
    };

    AbstractMatrix.prototype.asinh = function asinh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.asinh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.asinh = function asinh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.asinh();
    };

    AbstractMatrix.prototype.atan = function atan() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.atan(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.atan = function atan(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.atan();
    };

    AbstractMatrix.prototype.atanh = function atanh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.atanh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.atanh = function atanh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.atanh();
    };

    AbstractMatrix.prototype.cbrt = function cbrt() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.cbrt(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.cbrt = function cbrt(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.cbrt();
    };

    AbstractMatrix.prototype.ceil = function ceil() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.ceil(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.ceil = function ceil(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.ceil();
    };

    AbstractMatrix.prototype.clz32 = function clz32() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.clz32(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.clz32 = function clz32(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.clz32();
    };

    AbstractMatrix.prototype.cos = function cos() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.cos(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.cos = function cos(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.cos();
    };

    AbstractMatrix.prototype.cosh = function cosh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.cosh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.cosh = function cosh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.cosh();
    };

    AbstractMatrix.prototype.exp = function exp() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.exp(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.exp = function exp(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.exp();
    };

    AbstractMatrix.prototype.expm1 = function expm1() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.expm1(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.expm1 = function expm1(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.expm1();
    };

    AbstractMatrix.prototype.floor = function floor() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.floor(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.floor = function floor(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.floor();
    };

    AbstractMatrix.prototype.fround = function fround() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.fround(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.fround = function fround(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.fround();
    };

    AbstractMatrix.prototype.log = function log() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.log(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.log = function log(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.log();
    };

    AbstractMatrix.prototype.log1p = function log1p() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.log1p(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.log1p = function log1p(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.log1p();
    };

    AbstractMatrix.prototype.log10 = function log10() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.log10(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.log10 = function log10(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.log10();
    };

    AbstractMatrix.prototype.log2 = function log2() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.log2(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.log2 = function log2(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.log2();
    };

    AbstractMatrix.prototype.round = function round() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.round(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.round = function round(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.round();
    };

    AbstractMatrix.prototype.sign = function sign() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.sign(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.sign = function sign(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.sign();
    };

    AbstractMatrix.prototype.sin = function sin() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.sin(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.sin = function sin(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.sin();
    };

    AbstractMatrix.prototype.sinh = function sinh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.sinh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.sinh = function sinh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.sinh();
    };

    AbstractMatrix.prototype.sqrt = function sqrt() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.sqrt(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.sqrt = function sqrt(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.sqrt();
    };

    AbstractMatrix.prototype.tan = function tan() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.tan(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.tan = function tan(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.tan();
    };

    AbstractMatrix.prototype.tanh = function tanh() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.tanh(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.tanh = function tanh(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.tanh();
    };

    AbstractMatrix.prototype.trunc = function trunc() {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.trunc(this.get(i, j)));
        }
      }

      return this;
    };

    AbstractMatrix.trunc = function trunc(matrix) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.trunc();
    };

    AbstractMatrix.pow = function pow(matrix, arg0) {
      const newMatrix = new Matrix(matrix);
      return newMatrix.pow(arg0);
    };

    AbstractMatrix.prototype.pow = function pow(value) {
      if (typeof value === 'number') return this.powS(value);
      return this.powM(value);
    };

    AbstractMatrix.prototype.powS = function powS(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.pow(this.get(i, j), value));
        }
      }

      return this;
    };

    AbstractMatrix.prototype.powM = function powM(matrix) {
      matrix = Matrix.checkMatrix(matrix);

      if (this.rows !== matrix.rows || this.columns !== matrix.columns) {
        throw new RangeError('Matrices dimensions must be equal');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, Math.pow(this.get(i, j), matrix.get(i, j)));
        }
      }

      return this;
    };
  }

  class AbstractMatrix {
    static from1DArray(newRows, newColumns, newData) {
      let length = newRows * newColumns;

      if (length !== newData.length) {
        throw new RangeError('data length does not match given dimensions');
      }

      let newMatrix = new Matrix(newRows, newColumns);

      for (let row = 0; row < newRows; row++) {
        for (let column = 0; column < newColumns; column++) {
          newMatrix.set(row, column, newData[row * newColumns + column]);
        }
      }

      return newMatrix;
    }

    static rowVector(newData) {
      let vector = new Matrix(1, newData.length);

      for (let i = 0; i < newData.length; i++) {
        vector.set(0, i, newData[i]);
      }

      return vector;
    }

    static columnVector(newData) {
      let vector = new Matrix(newData.length, 1);

      for (let i = 0; i < newData.length; i++) {
        vector.set(i, 0, newData[i]);
      }

      return vector;
    }

    static zeros(rows, columns) {
      return new Matrix(rows, columns);
    }

    static ones(rows, columns) {
      return new Matrix(rows, columns).fill(1);
    }

    static rand(rows, columns) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        random = Math.random
      } = options;
      let matrix = new Matrix(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          matrix.set(i, j, random());
        }
      }

      return matrix;
    }

    static randInt(rows, columns) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        min = 0,
        max = 1000,
        random = Math.random
      } = options;
      if (!Number.isInteger(min)) throw new TypeError('min must be an integer');
      if (!Number.isInteger(max)) throw new TypeError('max must be an integer');
      if (min >= max) throw new RangeError('min must be smaller than max');
      let interval = max - min;
      let matrix = new Matrix(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          let value = min + Math.round(random() * interval);
          matrix.set(i, j, value);
        }
      }

      return matrix;
    }

    static eye(rows, columns, value) {
      if (columns === undefined) columns = rows;
      if (value === undefined) value = 1;
      let min = Math.min(rows, columns);
      let matrix = this.zeros(rows, columns);

      for (let i = 0; i < min; i++) {
        matrix.set(i, i, value);
      }

      return matrix;
    }

    static diag(data, rows, columns) {
      let l = data.length;
      if (rows === undefined) rows = l;
      if (columns === undefined) columns = rows;
      let min = Math.min(l, rows, columns);
      let matrix = this.zeros(rows, columns);

      for (let i = 0; i < min; i++) {
        matrix.set(i, i, data[i]);
      }

      return matrix;
    }

    static min(matrix1, matrix2) {
      matrix1 = this.checkMatrix(matrix1);
      matrix2 = this.checkMatrix(matrix2);
      let rows = matrix1.rows;
      let columns = matrix1.columns;
      let result = new Matrix(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          result.set(i, j, Math.min(matrix1.get(i, j), matrix2.get(i, j)));
        }
      }

      return result;
    }

    static max(matrix1, matrix2) {
      matrix1 = this.checkMatrix(matrix1);
      matrix2 = this.checkMatrix(matrix2);
      let rows = matrix1.rows;
      let columns = matrix1.columns;
      let result = new this(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          result.set(i, j, Math.max(matrix1.get(i, j), matrix2.get(i, j)));
        }
      }

      return result;
    }

    static checkMatrix(value) {
      return AbstractMatrix.isMatrix(value) ? value : new Matrix(value);
    }

    static isMatrix(value) {
      return value != null && value.klass === 'Matrix';
    }

    get size() {
      return this.rows * this.columns;
    }

    apply(callback) {
      if (typeof callback !== 'function') {
        throw new TypeError('callback must be a function');
      }

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          callback.call(this, i, j);
        }
      }

      return this;
    }

    to1DArray() {
      let array = [];

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          array.push(this.get(i, j));
        }
      }

      return array;
    }

    to2DArray() {
      let copy = [];

      for (let i = 0; i < this.rows; i++) {
        copy.push([]);

        for (let j = 0; j < this.columns; j++) {
          copy[i].push(this.get(i, j));
        }
      }

      return copy;
    }

    toJSON() {
      return this.to2DArray();
    }

    isRowVector() {
      return this.rows === 1;
    }

    isColumnVector() {
      return this.columns === 1;
    }

    isVector() {
      return this.rows === 1 || this.columns === 1;
    }

    isSquare() {
      return this.rows === this.columns;
    }

    isSymmetric() {
      if (this.isSquare()) {
        for (let i = 0; i < this.rows; i++) {
          for (let j = 0; j <= i; j++) {
            if (this.get(i, j) !== this.get(j, i)) {
              return false;
            }
          }
        }

        return true;
      }

      return false;
    }

    isEchelonForm() {
      let i = 0;
      let j = 0;
      let previousColumn = -1;
      let isEchelonForm = true;
      let checked = false;

      while (i < this.rows && isEchelonForm) {
        j = 0;
        checked = false;

        while (j < this.columns && checked === false) {
          if (this.get(i, j) === 0) {
            j++;
          } else if (this.get(i, j) === 1 && j > previousColumn) {
            checked = true;
            previousColumn = j;
          } else {
            isEchelonForm = false;
            checked = true;
          }
        }

        i++;
      }

      return isEchelonForm;
    }

    isReducedEchelonForm() {
      let i = 0;
      let j = 0;
      let previousColumn = -1;
      let isReducedEchelonForm = true;
      let checked = false;

      while (i < this.rows && isReducedEchelonForm) {
        j = 0;
        checked = false;

        while (j < this.columns && checked === false) {
          if (this.get(i, j) === 0) {
            j++;
          } else if (this.get(i, j) === 1 && j > previousColumn) {
            checked = true;
            previousColumn = j;
          } else {
            isReducedEchelonForm = false;
            checked = true;
          }
        }

        for (let k = j + 1; k < this.rows; k++) {
          if (this.get(i, k) !== 0) {
            isReducedEchelonForm = false;
          }
        }

        i++;
      }

      return isReducedEchelonForm;
    }

    echelonForm() {
      let result = this.clone();
      let h = 0;
      let k = 0;

      while (h < result.rows && k < result.columns) {
        let iMax = h;

        for (let i = h; i < result.rows; i++) {
          if (result.get(i, k) > result.get(iMax, k)) {
            iMax = i;
          }
        }

        if (result.get(iMax, k) === 0) {
          k++;
        } else {
          result.swapRows(h, iMax);
          let tmp = result.get(h, k);

          for (let j = k; j < result.columns; j++) {
            result.set(h, j, result.get(h, j) / tmp);
          }

          for (let i = h + 1; i < result.rows; i++) {
            let factor = result.get(i, k) / result.get(h, k);
            result.set(i, k, 0);

            for (let j = k + 1; j < result.columns; j++) {
              result.set(i, j, result.get(i, j) - result.get(h, j) * factor);
            }
          }

          h++;
          k++;
        }
      }

      return result;
    }

    reducedEchelonForm() {
      let result = this.echelonForm();
      let m = result.columns;
      let n = result.rows;
      let h = n - 1;

      while (h >= 0) {
        if (result.maxRow(h) === 0) {
          h--;
        } else {
          let p = 0;
          let pivot = false;

          while (p < n && pivot === false) {
            if (result.get(h, p) === 1) {
              pivot = true;
            } else {
              p++;
            }
          }

          for (let i = 0; i < h; i++) {
            let factor = result.get(i, p);

            for (let j = p; j < m; j++) {
              let tmp = result.get(i, j) - factor * result.get(h, j);
              result.set(i, j, tmp);
            }
          }

          h--;
        }
      }

      return result;
    }

    set() {
      throw new Error('set method is unimplemented');
    }

    get() {
      throw new Error('get method is unimplemented');
    }

    repeat() {
      let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        rows = 1,
        columns = 1
      } = options;

      if (!Number.isInteger(rows) || rows <= 0) {
        throw new TypeError('rows must be a positive integer');
      }

      if (!Number.isInteger(columns) || columns <= 0) {
        throw new TypeError('columns must be a positive integer');
      }

      let matrix = new Matrix(this.rows * rows, this.columns * columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          matrix.setSubMatrix(this, this.rows * i, this.columns * j);
        }
      }

      return matrix;
    }

    fill(value) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, value);
        }
      }

      return this;
    }

    neg() {
      return this.mulS(-1);
    }

    getRow(index) {
      checkRowIndex(this, index);
      let row = [];

      for (let i = 0; i < this.columns; i++) {
        row.push(this.get(index, i));
      }

      return row;
    }

    getRowVector(index) {
      return Matrix.rowVector(this.getRow(index));
    }

    setRow(index, array) {
      checkRowIndex(this, index);
      array = checkRowVector(this, array);

      for (let i = 0; i < this.columns; i++) {
        this.set(index, i, array[i]);
      }

      return this;
    }

    swapRows(row1, row2) {
      checkRowIndex(this, row1);
      checkRowIndex(this, row2);

      for (let i = 0; i < this.columns; i++) {
        let temp = this.get(row1, i);
        this.set(row1, i, this.get(row2, i));
        this.set(row2, i, temp);
      }

      return this;
    }

    getColumn(index) {
      checkColumnIndex(this, index);
      let column = [];

      for (let i = 0; i < this.rows; i++) {
        column.push(this.get(i, index));
      }

      return column;
    }

    getColumnVector(index) {
      return Matrix.columnVector(this.getColumn(index));
    }

    setColumn(index, array) {
      checkColumnIndex(this, index);
      array = checkColumnVector(this, array);

      for (let i = 0; i < this.rows; i++) {
        this.set(i, index, array[i]);
      }

      return this;
    }

    swapColumns(column1, column2) {
      checkColumnIndex(this, column1);
      checkColumnIndex(this, column2);

      for (let i = 0; i < this.rows; i++) {
        let temp = this.get(i, column1);
        this.set(i, column1, this.get(i, column2));
        this.set(i, column2, temp);
      }

      return this;
    }

    addRowVector(vector) {
      vector = checkRowVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) + vector[j]);
        }
      }

      return this;
    }

    subRowVector(vector) {
      vector = checkRowVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) - vector[j]);
        }
      }

      return this;
    }

    mulRowVector(vector) {
      vector = checkRowVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) * vector[j]);
        }
      }

      return this;
    }

    divRowVector(vector) {
      vector = checkRowVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) / vector[j]);
        }
      }

      return this;
    }

    addColumnVector(vector) {
      vector = checkColumnVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) + vector[i]);
        }
      }

      return this;
    }

    subColumnVector(vector) {
      vector = checkColumnVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) - vector[i]);
        }
      }

      return this;
    }

    mulColumnVector(vector) {
      vector = checkColumnVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) * vector[i]);
        }
      }

      return this;
    }

    divColumnVector(vector) {
      vector = checkColumnVector(this, vector);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.set(i, j, this.get(i, j) / vector[i]);
        }
      }

      return this;
    }

    mulRow(index, value) {
      checkRowIndex(this, index);

      for (let i = 0; i < this.columns; i++) {
        this.set(index, i, this.get(index, i) * value);
      }

      return this;
    }

    mulColumn(index, value) {
      checkColumnIndex(this, index);

      for (let i = 0; i < this.rows; i++) {
        this.set(i, index, this.get(i, index) * value);
      }

      return this;
    }

    max() {
      let v = this.get(0, 0);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          if (this.get(i, j) > v) {
            v = this.get(i, j);
          }
        }
      }

      return v;
    }

    maxIndex() {
      let v = this.get(0, 0);
      let idx = [0, 0];

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          if (this.get(i, j) > v) {
            v = this.get(i, j);
            idx[0] = i;
            idx[1] = j;
          }
        }
      }

      return idx;
    }

    min() {
      let v = this.get(0, 0);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          if (this.get(i, j) < v) {
            v = this.get(i, j);
          }
        }
      }

      return v;
    }

    minIndex() {
      let v = this.get(0, 0);
      let idx = [0, 0];

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          if (this.get(i, j) < v) {
            v = this.get(i, j);
            idx[0] = i;
            idx[1] = j;
          }
        }
      }

      return idx;
    }

    maxRow(row) {
      checkRowIndex(this, row);
      let v = this.get(row, 0);

      for (let i = 1; i < this.columns; i++) {
        if (this.get(row, i) > v) {
          v = this.get(row, i);
        }
      }

      return v;
    }

    maxRowIndex(row) {
      checkRowIndex(this, row);
      let v = this.get(row, 0);
      let idx = [row, 0];

      for (let i = 1; i < this.columns; i++) {
        if (this.get(row, i) > v) {
          v = this.get(row, i);
          idx[1] = i;
        }
      }

      return idx;
    }

    minRow(row) {
      checkRowIndex(this, row);
      let v = this.get(row, 0);

      for (let i = 1; i < this.columns; i++) {
        if (this.get(row, i) < v) {
          v = this.get(row, i);
        }
      }

      return v;
    }

    minRowIndex(row) {
      checkRowIndex(this, row);
      let v = this.get(row, 0);
      let idx = [row, 0];

      for (let i = 1; i < this.columns; i++) {
        if (this.get(row, i) < v) {
          v = this.get(row, i);
          idx[1] = i;
        }
      }

      return idx;
    }

    maxColumn(column) {
      checkColumnIndex(this, column);
      let v = this.get(0, column);

      for (let i = 1; i < this.rows; i++) {
        if (this.get(i, column) > v) {
          v = this.get(i, column);
        }
      }

      return v;
    }

    maxColumnIndex(column) {
      checkColumnIndex(this, column);
      let v = this.get(0, column);
      let idx = [0, column];

      for (let i = 1; i < this.rows; i++) {
        if (this.get(i, column) > v) {
          v = this.get(i, column);
          idx[0] = i;
        }
      }

      return idx;
    }

    minColumn(column) {
      checkColumnIndex(this, column);
      let v = this.get(0, column);

      for (let i = 1; i < this.rows; i++) {
        if (this.get(i, column) < v) {
          v = this.get(i, column);
        }
      }

      return v;
    }

    minColumnIndex(column) {
      checkColumnIndex(this, column);
      let v = this.get(0, column);
      let idx = [0, column];

      for (let i = 1; i < this.rows; i++) {
        if (this.get(i, column) < v) {
          v = this.get(i, column);
          idx[0] = i;
        }
      }

      return idx;
    }

    diag() {
      let min = Math.min(this.rows, this.columns);
      let diag = [];

      for (let i = 0; i < min; i++) {
        diag.push(this.get(i, i));
      }

      return diag;
    }

    norm() {
      let type = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 'frobenius';
      let result = 0;

      if (type === 'max') {
        return this.max();
      } else if (type === 'frobenius') {
        for (let i = 0; i < this.rows; i++) {
          for (let j = 0; j < this.columns; j++) {
            result = result + this.get(i, j) * this.get(i, j);
          }
        }

        return Math.sqrt(result);
      } else {
        throw new RangeError("unknown norm type: ".concat(type));
      }
    }

    cumulativeSum() {
      let sum = 0;

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          sum += this.get(i, j);
          this.set(i, j, sum);
        }
      }

      return this;
    }

    dot(vector2) {
      if (AbstractMatrix.isMatrix(vector2)) vector2 = vector2.to1DArray();
      let vector1 = this.to1DArray();

      if (vector1.length !== vector2.length) {
        throw new RangeError('vectors do not have the same size');
      }

      let dot = 0;

      for (let i = 0; i < vector1.length; i++) {
        dot += vector1[i] * vector2[i];
      }

      return dot;
    }

    mmul(other) {
      other = Matrix.checkMatrix(other);
      let m = this.rows;
      let n = this.columns;
      let p = other.columns;
      let result = new Matrix(m, p);
      let Bcolj = new Float64Array(n);

      for (let j = 0; j < p; j++) {
        for (let k = 0; k < n; k++) {
          Bcolj[k] = other.get(k, j);
        }

        for (let i = 0; i < m; i++) {
          let s = 0;

          for (let k = 0; k < n; k++) {
            s += this.get(i, k) * Bcolj[k];
          }

          result.set(i, j, s);
        }
      }

      return result;
    }

    strassen2x2(other) {
      other = Matrix.checkMatrix(other);
      let result = new Matrix(2, 2);
      const a11 = this.get(0, 0);
      const b11 = other.get(0, 0);
      const a12 = this.get(0, 1);
      const b12 = other.get(0, 1);
      const a21 = this.get(1, 0);
      const b21 = other.get(1, 0);
      const a22 = this.get(1, 1);
      const b22 = other.get(1, 1); // Compute intermediate values.

      const m1 = (a11 + a22) * (b11 + b22);
      const m2 = (a21 + a22) * b11;
      const m3 = a11 * (b12 - b22);
      const m4 = a22 * (b21 - b11);
      const m5 = (a11 + a12) * b22;
      const m6 = (a21 - a11) * (b11 + b12);
      const m7 = (a12 - a22) * (b21 + b22); // Combine intermediate values into the output.

      const c00 = m1 + m4 - m5 + m7;
      const c01 = m3 + m5;
      const c10 = m2 + m4;
      const c11 = m1 - m2 + m3 + m6;
      result.set(0, 0, c00);
      result.set(0, 1, c01);
      result.set(1, 0, c10);
      result.set(1, 1, c11);
      return result;
    }

    strassen3x3(other) {
      other = Matrix.checkMatrix(other);
      let result = new Matrix(3, 3);
      const a00 = this.get(0, 0);
      const a01 = this.get(0, 1);
      const a02 = this.get(0, 2);
      const a10 = this.get(1, 0);
      const a11 = this.get(1, 1);
      const a12 = this.get(1, 2);
      const a20 = this.get(2, 0);
      const a21 = this.get(2, 1);
      const a22 = this.get(2, 2);
      const b00 = other.get(0, 0);
      const b01 = other.get(0, 1);
      const b02 = other.get(0, 2);
      const b10 = other.get(1, 0);
      const b11 = other.get(1, 1);
      const b12 = other.get(1, 2);
      const b20 = other.get(2, 0);
      const b21 = other.get(2, 1);
      const b22 = other.get(2, 2);
      const m1 = (a00 + a01 + a02 - a10 - a11 - a21 - a22) * b11;
      const m2 = (a00 - a10) * (-b01 + b11);
      const m3 = a11 * (-b00 + b01 + b10 - b11 - b12 - b20 + b22);
      const m4 = (-a00 + a10 + a11) * (b00 - b01 + b11);
      const m5 = (a10 + a11) * (-b00 + b01);
      const m6 = a00 * b00;
      const m7 = (-a00 + a20 + a21) * (b00 - b02 + b12);
      const m8 = (-a00 + a20) * (b02 - b12);
      const m9 = (a20 + a21) * (-b00 + b02);
      const m10 = (a00 + a01 + a02 - a11 - a12 - a20 - a21) * b12;
      const m11 = a21 * (-b00 + b02 + b10 - b11 - b12 - b20 + b21);
      const m12 = (-a02 + a21 + a22) * (b11 + b20 - b21);
      const m13 = (a02 - a22) * (b11 - b21);
      const m14 = a02 * b20;
      const m15 = (a21 + a22) * (-b20 + b21);
      const m16 = (-a02 + a11 + a12) * (b12 + b20 - b22);
      const m17 = (a02 - a12) * (b12 - b22);
      const m18 = (a11 + a12) * (-b20 + b22);
      const m19 = a01 * b10;
      const m20 = a12 * b21;
      const m21 = a10 * b02;
      const m22 = a20 * b01;
      const m23 = a22 * b22;
      const c00 = m6 + m14 + m19;
      const c01 = m1 + m4 + m5 + m6 + m12 + m14 + m15;
      const c02 = m6 + m7 + m9 + m10 + m14 + m16 + m18;
      const c10 = m2 + m3 + m4 + m6 + m14 + m16 + m17;
      const c11 = m2 + m4 + m5 + m6 + m20;
      const c12 = m14 + m16 + m17 + m18 + m21;
      const c20 = m6 + m7 + m8 + m11 + m12 + m13 + m14;
      const c21 = m12 + m13 + m14 + m15 + m22;
      const c22 = m6 + m7 + m8 + m9 + m23;
      result.set(0, 0, c00);
      result.set(0, 1, c01);
      result.set(0, 2, c02);
      result.set(1, 0, c10);
      result.set(1, 1, c11);
      result.set(1, 2, c12);
      result.set(2, 0, c20);
      result.set(2, 1, c21);
      result.set(2, 2, c22);
      return result;
    }

    mmulStrassen(y) {
      y = Matrix.checkMatrix(y);
      let x = this.clone();
      let r1 = x.rows;
      let c1 = x.columns;
      let r2 = y.rows;
      let c2 = y.columns;

      if (c1 !== r2) {
        // eslint-disable-next-line no-console
        console.warn("Multiplying ".concat(r1, " x ").concat(c1, " and ").concat(r2, " x ").concat(c2, " matrix: dimensions do not match."));
      } // Put a matrix into the top left of a matrix of zeros.
      // `rows` and `cols` are the dimensions of the output matrix.


      function embed(mat, rows, cols) {
        let r = mat.rows;
        let c = mat.columns;

        if (r === rows && c === cols) {
          return mat;
        } else {
          let resultat = AbstractMatrix.zeros(rows, cols);
          resultat = resultat.setSubMatrix(mat, 0, 0);
          return resultat;
        }
      } // Make sure both matrices are the same size.
      // This is exclusively for simplicity:
      // this algorithm can be implemented with matrices of different sizes.


      let r = Math.max(r1, r2);
      let c = Math.max(c1, c2);
      x = embed(x, r, c);
      y = embed(y, r, c); // Our recursive multiplication function.

      function blockMult(a, b, rows, cols) {
        // For small matrices, resort to naive multiplication.
        if (rows <= 512 || cols <= 512) {
          return a.mmul(b); // a is equivalent to this
        } // Apply dynamic padding.


        if (rows % 2 === 1 && cols % 2 === 1) {
          a = embed(a, rows + 1, cols + 1);
          b = embed(b, rows + 1, cols + 1);
        } else if (rows % 2 === 1) {
          a = embed(a, rows + 1, cols);
          b = embed(b, rows + 1, cols);
        } else if (cols % 2 === 1) {
          a = embed(a, rows, cols + 1);
          b = embed(b, rows, cols + 1);
        }

        let halfRows = parseInt(a.rows / 2, 10);
        let halfCols = parseInt(a.columns / 2, 10); // Subdivide input matrices.

        let a11 = a.subMatrix(0, halfRows - 1, 0, halfCols - 1);
        let b11 = b.subMatrix(0, halfRows - 1, 0, halfCols - 1);
        let a12 = a.subMatrix(0, halfRows - 1, halfCols, a.columns - 1);
        let b12 = b.subMatrix(0, halfRows - 1, halfCols, b.columns - 1);
        let a21 = a.subMatrix(halfRows, a.rows - 1, 0, halfCols - 1);
        let b21 = b.subMatrix(halfRows, b.rows - 1, 0, halfCols - 1);
        let a22 = a.subMatrix(halfRows, a.rows - 1, halfCols, a.columns - 1);
        let b22 = b.subMatrix(halfRows, b.rows - 1, halfCols, b.columns - 1); // Compute intermediate values.

        let m1 = blockMult(AbstractMatrix.add(a11, a22), AbstractMatrix.add(b11, b22), halfRows, halfCols);
        let m2 = blockMult(AbstractMatrix.add(a21, a22), b11, halfRows, halfCols);
        let m3 = blockMult(a11, AbstractMatrix.sub(b12, b22), halfRows, halfCols);
        let m4 = blockMult(a22, AbstractMatrix.sub(b21, b11), halfRows, halfCols);
        let m5 = blockMult(AbstractMatrix.add(a11, a12), b22, halfRows, halfCols);
        let m6 = blockMult(AbstractMatrix.sub(a21, a11), AbstractMatrix.add(b11, b12), halfRows, halfCols);
        let m7 = blockMult(AbstractMatrix.sub(a12, a22), AbstractMatrix.add(b21, b22), halfRows, halfCols); // Combine intermediate values into the output.

        let c11 = AbstractMatrix.add(m1, m4);
        c11.sub(m5);
        c11.add(m7);
        let c12 = AbstractMatrix.add(m3, m5);
        let c21 = AbstractMatrix.add(m2, m4);
        let c22 = AbstractMatrix.sub(m1, m2);
        c22.add(m3);
        c22.add(m6); // Crop output to the desired size (undo dynamic padding).

        let resultat = AbstractMatrix.zeros(2 * c11.rows, 2 * c11.columns);
        resultat = resultat.setSubMatrix(c11, 0, 0);
        resultat = resultat.setSubMatrix(c12, c11.rows, 0);
        resultat = resultat.setSubMatrix(c21, 0, c11.columns);
        resultat = resultat.setSubMatrix(c22, c11.rows, c11.columns);
        return resultat.subMatrix(0, rows - 1, 0, cols - 1);
      }

      return blockMult(x, y, r, c);
    }

    scaleRows() {
      let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        min = 0,
        max = 1
      } = options;
      if (!Number.isFinite(min)) throw new TypeError('min must be a number');
      if (!Number.isFinite(max)) throw new TypeError('max must be a number');
      if (min >= max) throw new RangeError('min must be smaller than max');
      let newMatrix = new Matrix(this.rows, this.columns);

      for (let i = 0; i < this.rows; i++) {
        const row = this.getRow(i);
        rescale(row, {
          min,
          max,
          output: row
        });
        newMatrix.setRow(i, row);
      }

      return newMatrix;
    }

    scaleColumns() {
      let options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        min = 0,
        max = 1
      } = options;
      if (!Number.isFinite(min)) throw new TypeError('min must be a number');
      if (!Number.isFinite(max)) throw new TypeError('max must be a number');
      if (min >= max) throw new RangeError('min must be smaller than max');
      let newMatrix = new Matrix(this.rows, this.columns);

      for (let i = 0; i < this.columns; i++) {
        const column = this.getColumn(i);
        rescale(column, {
          min: min,
          max: max,
          output: column
        });
        newMatrix.setColumn(i, column);
      }

      return newMatrix;
    }

    flipRows() {
      const middle = Math.ceil(this.columns / 2);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < middle; j++) {
          let first = this.get(i, j);
          let last = this.get(i, this.columns - 1 - j);
          this.set(i, j, last);
          this.set(i, this.columns - 1 - j, first);
        }
      }

      return this;
    }

    flipColumns() {
      const middle = Math.ceil(this.rows / 2);

      for (let j = 0; j < this.columns; j++) {
        for (let i = 0; i < middle; i++) {
          let first = this.get(i, j);
          let last = this.get(this.rows - 1 - i, j);
          this.set(i, j, last);
          this.set(this.rows - 1 - i, j, first);
        }
      }

      return this;
    }

    kroneckerProduct(other) {
      other = Matrix.checkMatrix(other);
      let m = this.rows;
      let n = this.columns;
      let p = other.rows;
      let q = other.columns;
      let result = new Matrix(m * p, n * q);

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < p; k++) {
            for (let l = 0; l < q; l++) {
              result.set(p * i + k, q * j + l, this.get(i, j) * other.get(k, l));
            }
          }
        }
      }

      return result;
    }

    transpose() {
      let result = new Matrix(this.columns, this.rows);

      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          result.set(j, i, this.get(i, j));
        }
      }

      return result;
    }

    sortRows() {
      let compareFunction = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : compareNumbers;

      for (let i = 0; i < this.rows; i++) {
        this.setRow(i, this.getRow(i).sort(compareFunction));
      }

      return this;
    }

    sortColumns() {
      let compareFunction = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : compareNumbers;

      for (let i = 0; i < this.columns; i++) {
        this.setColumn(i, this.getColumn(i).sort(compareFunction));
      }

      return this;
    }

    subMatrix(startRow, endRow, startColumn, endColumn) {
      checkRange(this, startRow, endRow, startColumn, endColumn);
      let newMatrix = new Matrix(endRow - startRow + 1, endColumn - startColumn + 1);

      for (let i = startRow; i <= endRow; i++) {
        for (let j = startColumn; j <= endColumn; j++) {
          newMatrix.set(i - startRow, j - startColumn, this.get(i, j));
        }
      }

      return newMatrix;
    }

    subMatrixRow(indices, startColumn, endColumn) {
      if (startColumn === undefined) startColumn = 0;
      if (endColumn === undefined) endColumn = this.columns - 1;

      if (startColumn > endColumn || startColumn < 0 || startColumn >= this.columns || endColumn < 0 || endColumn >= this.columns) {
        throw new RangeError('Argument out of range');
      }

      let newMatrix = new Matrix(indices.length, endColumn - startColumn + 1);

      for (let i = 0; i < indices.length; i++) {
        for (let j = startColumn; j <= endColumn; j++) {
          if (indices[i] < 0 || indices[i] >= this.rows) {
            throw new RangeError("Row index out of range: ".concat(indices[i]));
          }

          newMatrix.set(i, j - startColumn, this.get(indices[i], j));
        }
      }

      return newMatrix;
    }

    subMatrixColumn(indices, startRow, endRow) {
      if (startRow === undefined) startRow = 0;
      if (endRow === undefined) endRow = this.rows - 1;

      if (startRow > endRow || startRow < 0 || startRow >= this.rows || endRow < 0 || endRow >= this.rows) {
        throw new RangeError('Argument out of range');
      }

      let newMatrix = new Matrix(endRow - startRow + 1, indices.length);

      for (let i = 0; i < indices.length; i++) {
        for (let j = startRow; j <= endRow; j++) {
          if (indices[i] < 0 || indices[i] >= this.columns) {
            throw new RangeError("Column index out of range: ".concat(indices[i]));
          }

          newMatrix.set(j - startRow, i, this.get(j, indices[i]));
        }
      }

      return newMatrix;
    }

    setSubMatrix(matrix, startRow, startColumn) {
      matrix = Matrix.checkMatrix(matrix);
      let endRow = startRow + matrix.rows - 1;
      let endColumn = startColumn + matrix.columns - 1;
      checkRange(this, startRow, endRow, startColumn, endColumn);

      for (let i = 0; i < matrix.rows; i++) {
        for (let j = 0; j < matrix.columns; j++) {
          this.set(startRow + i, startColumn + j, matrix.get(i, j));
        }
      }

      return this;
    }

    selection(rowIndices, columnIndices) {
      let indices = checkIndices(this, rowIndices, columnIndices);
      let newMatrix = new Matrix(rowIndices.length, columnIndices.length);

      for (let i = 0; i < indices.row.length; i++) {
        let rowIndex = indices.row[i];

        for (let j = 0; j < indices.column.length; j++) {
          let columnIndex = indices.column[j];
          newMatrix.set(i, j, this.get(rowIndex, columnIndex));
        }
      }

      return newMatrix;
    }

    trace() {
      let min = Math.min(this.rows, this.columns);
      let trace = 0;

      for (let i = 0; i < min; i++) {
        trace += this.get(i, i);
      }

      return trace;
    }

    clone() {
      let newMatrix = new Matrix(this.rows, this.columns);

      for (let row = 0; row < this.rows; row++) {
        for (let column = 0; column < this.columns; column++) {
          newMatrix.set(row, column, this.get(row, column));
        }
      }

      return newMatrix;
    }

    sum(by) {
      switch (by) {
        case 'row':
          return sumByRow(this);

        case 'column':
          return sumByColumn(this);

        case undefined:
          return sumAll(this);

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

    product(by) {
      switch (by) {
        case 'row':
          return productByRow(this);

        case 'column':
          return productByColumn(this);

        case undefined:
          return productAll(this);

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

    mean(by) {
      const sum = this.sum(by);

      switch (by) {
        case 'row':
          {
            for (let i = 0; i < this.rows; i++) {
              sum[i] /= this.columns;
            }

            return sum;
          }

        case 'column':
          {
            for (let i = 0; i < this.columns; i++) {
              sum[i] /= this.rows;
            }

            return sum;
          }

        case undefined:
          return sum / this.size;

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

    variance(by) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      if (typeof by === 'object') {
        options = by;
        by = undefined;
      }

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        unbiased = true,
        mean = this.mean(by)
      } = options;

      if (typeof unbiased !== 'boolean') {
        throw new TypeError('unbiased must be a boolean');
      }

      switch (by) {
        case 'row':
          {
            if (!Array.isArray(mean)) {
              throw new TypeError('mean must be an array');
            }

            return varianceByRow(this, unbiased, mean);
          }

        case 'column':
          {
            if (!Array.isArray(mean)) {
              throw new TypeError('mean must be an array');
            }

            return varianceByColumn(this, unbiased, mean);
          }

        case undefined:
          {
            if (typeof mean !== 'number') {
              throw new TypeError('mean must be a number');
            }

            return varianceAll(this, unbiased, mean);
          }

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

    standardDeviation(by, options) {
      if (typeof by === 'object') {
        options = by;
        by = undefined;
      }

      const variance = this.variance(by, options);

      if (by === undefined) {
        return Math.sqrt(variance);
      } else {
        for (let i = 0; i < variance.length; i++) {
          variance[i] = Math.sqrt(variance[i]);
        }

        return variance;
      }
    }

    center(by) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      if (typeof by === 'object') {
        options = by;
        by = undefined;
      }

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      const {
        center = this.mean(by)
      } = options;

      switch (by) {
        case 'row':
          {
            if (!Array.isArray(center)) {
              throw new TypeError('center must be an array');
            }

            centerByRow(this, center);
            return this;
          }

        case 'column':
          {
            if (!Array.isArray(center)) {
              throw new TypeError('center must be an array');
            }

            centerByColumn(this, center);
            return this;
          }

        case undefined:
          {
            if (typeof center !== 'number') {
              throw new TypeError('center must be a number');
            }

            centerAll(this, center);
            return this;
          }

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

    scale(by) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      if (typeof by === 'object') {
        options = by;
        by = undefined;
      }

      if (typeof options !== 'object') {
        throw new TypeError('options must be an object');
      }

      let scale = options.scale;

      switch (by) {
        case 'row':
          {
            if (scale === undefined) {
              scale = getScaleByRow(this);
            } else if (!Array.isArray(scale)) {
              throw new TypeError('scale must be an array');
            }

            scaleByRow(this, scale);
            return this;
          }

        case 'column':
          {
            if (scale === undefined) {
              scale = getScaleByColumn(this);
            } else if (!Array.isArray(scale)) {
              throw new TypeError('scale must be an array');
            }

            scaleByColumn(this, scale);
            return this;
          }

        case undefined:
          {
            if (scale === undefined) {
              scale = getScaleAll(this);
            } else if (typeof scale !== 'number') {
              throw new TypeError('scale must be a number');
            }

            scaleAll(this, scale);
            return this;
          }

        default:
          throw new Error("invalid option: ".concat(by));
      }
    }

  }
  AbstractMatrix.prototype.klass = 'Matrix';

  if (typeof Symbol !== 'undefined') {
    AbstractMatrix.prototype[Symbol.for('nodejs.util.inspect.custom')] = inspectMatrix;
  }

  function compareNumbers(a, b) {
    return a - b;
  } // Synonyms


  AbstractMatrix.random = AbstractMatrix.rand;
  AbstractMatrix.randomInt = AbstractMatrix.randInt;
  AbstractMatrix.diagonal = AbstractMatrix.diag;
  AbstractMatrix.prototype.diagonal = AbstractMatrix.prototype.diag;
  AbstractMatrix.identity = AbstractMatrix.eye;
  AbstractMatrix.prototype.negate = AbstractMatrix.prototype.neg;
  AbstractMatrix.prototype.tensorProduct = AbstractMatrix.prototype.kroneckerProduct;
  class Matrix extends AbstractMatrix {
    constructor(nRows, nColumns) {
      super();

      if (Matrix.isMatrix(nRows)) {
        return nRows.clone();
      } else if (Number.isInteger(nRows) && nRows > 0) {
        // Create an empty matrix
        this.data = [];

        if (Number.isInteger(nColumns) && nColumns > 0) {
          for (let i = 0; i < nRows; i++) {
            this.data.push(new Float64Array(nColumns));
          }
        } else {
          throw new TypeError('nColumns must be a positive integer');
        }
      } else if (Array.isArray(nRows)) {
        // Copy the values from the 2D array
        const arrayData = nRows;
        nRows = arrayData.length;
        nColumns = arrayData[0].length;

        if (typeof nColumns !== 'number' || nColumns === 0) {
          throw new TypeError('Data must be a 2D array with at least one element');
        }

        this.data = [];

        for (let i = 0; i < nRows; i++) {
          if (arrayData[i].length !== nColumns) {
            throw new RangeError('Inconsistent array dimensions');
          }

          this.data.push(Float64Array.from(arrayData[i]));
        }
      } else {
        throw new TypeError('First argument must be a positive number or an array');
      }

      this.rows = nRows;
      this.columns = nColumns;
      return this;
    }

    set(rowIndex, columnIndex, value) {
      this.data[rowIndex][columnIndex] = value;
      return this;
    }

    get(rowIndex, columnIndex) {
      return this.data[rowIndex][columnIndex];
    }

    removeRow(index) {
      checkRowIndex(this, index);

      if (this.rows === 1) {
        throw new RangeError('A matrix cannot have less than one row');
      }

      this.data.splice(index, 1);
      this.rows -= 1;
      return this;
    }

    addRow(index, array) {
      if (array === undefined) {
        array = index;
        index = this.rows;
      }

      checkRowIndex(this, index, true);
      array = Float64Array.from(checkRowVector(this, array));
      this.data.splice(index, 0, array);
      this.rows += 1;
      return this;
    }

    removeColumn(index) {
      checkColumnIndex(this, index);

      if (this.columns === 1) {
        throw new RangeError('A matrix cannot have less than one column');
      }

      for (let i = 0; i < this.rows; i++) {
        const newRow = new Float64Array(this.columns - 1);

        for (let j = 0; j < index; j++) {
          newRow[j] = this.data[i][j];
        }

        for (let j = index + 1; j < this.columns; j++) {
          newRow[j - 1] = this.data[i][j];
        }

        this.data[i] = newRow;
      }

      this.columns -= 1;
      return this;
    }

    addColumn(index, array) {
      if (typeof array === 'undefined') {
        array = index;
        index = this.columns;
      }

      checkColumnIndex(this, index, true);
      array = checkColumnVector(this, array);

      for (let i = 0; i < this.rows; i++) {
        const newRow = new Float64Array(this.columns + 1);
        let j = 0;

        for (; j < index; j++) {
          newRow[j] = this.data[i][j];
        }

        newRow[j++] = array[i];

        for (; j < this.columns + 1; j++) {
          newRow[j] = this.data[i][j - 1];
        }

        this.data[i] = newRow;
      }

      this.columns += 1;
      return this;
    }

  }
  installMathOperations(AbstractMatrix, Matrix);

  class WrapperMatrix2D extends AbstractMatrix {
    constructor(data) {
      super();
      this.data = data;
      this.rows = data.length;
      this.columns = data[0].length;
    }

    set(rowIndex, columnIndex, value) {
      this.data[rowIndex][columnIndex] = value;
      return this;
    }

    get(rowIndex, columnIndex) {
      return this.data[rowIndex][columnIndex];
    }

  }

  class LuDecomposition {
    constructor(matrix) {
      matrix = WrapperMatrix2D.checkMatrix(matrix);
      let lu = matrix.clone();
      let rows = lu.rows;
      let columns = lu.columns;
      let pivotVector = new Float64Array(rows);
      let pivotSign = 1;
      let i, j, k, p, s, t, v;
      let LUcolj, kmax;

      for (i = 0; i < rows; i++) {
        pivotVector[i] = i;
      }

      LUcolj = new Float64Array(rows);

      for (j = 0; j < columns; j++) {
        for (i = 0; i < rows; i++) {
          LUcolj[i] = lu.get(i, j);
        }

        for (i = 0; i < rows; i++) {
          kmax = Math.min(i, j);
          s = 0;

          for (k = 0; k < kmax; k++) {
            s += lu.get(i, k) * LUcolj[k];
          }

          LUcolj[i] -= s;
          lu.set(i, j, LUcolj[i]);
        }

        p = j;

        for (i = j + 1; i < rows; i++) {
          if (Math.abs(LUcolj[i]) > Math.abs(LUcolj[p])) {
            p = i;
          }
        }

        if (p !== j) {
          for (k = 0; k < columns; k++) {
            t = lu.get(p, k);
            lu.set(p, k, lu.get(j, k));
            lu.set(j, k, t);
          }

          v = pivotVector[p];
          pivotVector[p] = pivotVector[j];
          pivotVector[j] = v;
          pivotSign = -pivotSign;
        }

        if (j < rows && lu.get(j, j) !== 0) {
          for (i = j + 1; i < rows; i++) {
            lu.set(i, j, lu.get(i, j) / lu.get(j, j));
          }
        }
      }

      this.LU = lu;
      this.pivotVector = pivotVector;
      this.pivotSign = pivotSign;
    }

    isSingular() {
      let data = this.LU;
      let col = data.columns;

      for (let j = 0; j < col; j++) {
        if (data.get(j, j) === 0) {
          return true;
        }
      }

      return false;
    }

    solve(value) {
      value = Matrix.checkMatrix(value);
      let lu = this.LU;
      let rows = lu.rows;

      if (rows !== value.rows) {
        throw new Error('Invalid matrix dimensions');
      }

      if (this.isSingular()) {
        throw new Error('LU matrix is singular');
      }

      let count = value.columns;
      let X = value.subMatrixRow(this.pivotVector, 0, count - 1);
      let columns = lu.columns;
      let i, j, k;

      for (k = 0; k < columns; k++) {
        for (i = k + 1; i < columns; i++) {
          for (j = 0; j < count; j++) {
            X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
          }
        }
      }

      for (k = columns - 1; k >= 0; k--) {
        for (j = 0; j < count; j++) {
          X.set(k, j, X.get(k, j) / lu.get(k, k));
        }

        for (i = 0; i < k; i++) {
          for (j = 0; j < count; j++) {
            X.set(i, j, X.get(i, j) - X.get(k, j) * lu.get(i, k));
          }
        }
      }

      return X;
    }

    get determinant() {
      let data = this.LU;

      if (!data.isSquare()) {
        throw new Error('Matrix must be square');
      }

      let determinant = this.pivotSign;
      let col = data.columns;

      for (let j = 0; j < col; j++) {
        determinant *= data.get(j, j);
      }

      return determinant;
    }

    get lowerTriangularMatrix() {
      let data = this.LU;
      let rows = data.rows;
      let columns = data.columns;
      let X = new Matrix(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          if (i > j) {
            X.set(i, j, data.get(i, j));
          } else if (i === j) {
            X.set(i, j, 1);
          } else {
            X.set(i, j, 0);
          }
        }
      }

      return X;
    }

    get upperTriangularMatrix() {
      let data = this.LU;
      let rows = data.rows;
      let columns = data.columns;
      let X = new Matrix(rows, columns);

      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < columns; j++) {
          if (i <= j) {
            X.set(i, j, data.get(i, j));
          } else {
            X.set(i, j, 0);
          }
        }
      }

      return X;
    }

    get pivotPermutationVector() {
      return Array.from(this.pivotVector);
    }

  }

  function hypotenuse(a, b) {
    let r = 0;

    if (Math.abs(a) > Math.abs(b)) {
      r = b / a;
      return Math.abs(a) * Math.sqrt(1 + r * r);
    }

    if (b !== 0) {
      r = a / b;
      return Math.abs(b) * Math.sqrt(1 + r * r);
    }

    return 0;
  }

  class QrDecomposition {
    constructor(value) {
      value = WrapperMatrix2D.checkMatrix(value);
      let qr = value.clone();
      let m = value.rows;
      let n = value.columns;
      let rdiag = new Float64Array(n);
      let i, j, k, s;

      for (k = 0; k < n; k++) {
        let nrm = 0;

        for (i = k; i < m; i++) {
          nrm = hypotenuse(nrm, qr.get(i, k));
        }

        if (nrm !== 0) {
          if (qr.get(k, k) < 0) {
            nrm = -nrm;
          }

          for (i = k; i < m; i++) {
            qr.set(i, k, qr.get(i, k) / nrm);
          }

          qr.set(k, k, qr.get(k, k) + 1);

          for (j = k + 1; j < n; j++) {
            s = 0;

            for (i = k; i < m; i++) {
              s += qr.get(i, k) * qr.get(i, j);
            }

            s = -s / qr.get(k, k);

            for (i = k; i < m; i++) {
              qr.set(i, j, qr.get(i, j) + s * qr.get(i, k));
            }
          }
        }

        rdiag[k] = -nrm;
      }

      this.QR = qr;
      this.Rdiag = rdiag;
    }

    solve(value) {
      value = Matrix.checkMatrix(value);
      let qr = this.QR;
      let m = qr.rows;

      if (value.rows !== m) {
        throw new Error('Matrix row dimensions must agree');
      }

      if (!this.isFullRank()) {
        throw new Error('Matrix is rank deficient');
      }

      let count = value.columns;
      let X = value.clone();
      let n = qr.columns;
      let i, j, k, s;

      for (k = 0; k < n; k++) {
        for (j = 0; j < count; j++) {
          s = 0;

          for (i = k; i < m; i++) {
            s += qr.get(i, k) * X.get(i, j);
          }

          s = -s / qr.get(k, k);

          for (i = k; i < m; i++) {
            X.set(i, j, X.get(i, j) + s * qr.get(i, k));
          }
        }
      }

      for (k = n - 1; k >= 0; k--) {
        for (j = 0; j < count; j++) {
          X.set(k, j, X.get(k, j) / this.Rdiag[k]);
        }

        for (i = 0; i < k; i++) {
          for (j = 0; j < count; j++) {
            X.set(i, j, X.get(i, j) - X.get(k, j) * qr.get(i, k));
          }
        }
      }

      return X.subMatrix(0, n - 1, 0, count - 1);
    }

    isFullRank() {
      let columns = this.QR.columns;

      for (let i = 0; i < columns; i++) {
        if (this.Rdiag[i] === 0) {
          return false;
        }
      }

      return true;
    }

    get upperTriangularMatrix() {
      let qr = this.QR;
      let n = qr.columns;
      let X = new Matrix(n, n);
      let i, j;

      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          if (i < j) {
            X.set(i, j, qr.get(i, j));
          } else if (i === j) {
            X.set(i, j, this.Rdiag[i]);
          } else {
            X.set(i, j, 0);
          }
        }
      }

      return X;
    }

    get orthogonalMatrix() {
      let qr = this.QR;
      let rows = qr.rows;
      let columns = qr.columns;
      let X = new Matrix(rows, columns);
      let i, j, k, s;

      for (k = columns - 1; k >= 0; k--) {
        for (i = 0; i < rows; i++) {
          X.set(i, k, 0);
        }

        X.set(k, k, 1);

        for (j = k; j < columns; j++) {
          if (qr.get(k, k) !== 0) {
            s = 0;

            for (i = k; i < rows; i++) {
              s += qr.get(i, k) * X.get(i, j);
            }

            s = -s / qr.get(k, k);

            for (i = k; i < rows; i++) {
              X.set(i, j, X.get(i, j) + s * qr.get(i, k));
            }
          }
        }
      }

      return X;
    }

  }

  class SingularValueDecomposition {
    constructor(value) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
      value = WrapperMatrix2D.checkMatrix(value);
      let m = value.rows;
      let n = value.columns;
      const {
        computeLeftSingularVectors = true,
        computeRightSingularVectors = true,
        autoTranspose = false
      } = options;
      let wantu = Boolean(computeLeftSingularVectors);
      let wantv = Boolean(computeRightSingularVectors);
      let swapped = false;
      let a;

      if (m < n) {
        if (!autoTranspose) {
          a = value.clone(); // eslint-disable-next-line no-console

          console.warn('Computing SVD on a matrix with more columns than rows. Consider enabling autoTranspose');
        } else {
          a = value.transpose();
          m = a.rows;
          n = a.columns;
          swapped = true;
          let aux = wantu;
          wantu = wantv;
          wantv = aux;
        }
      } else {
        a = value.clone();
      }

      let nu = Math.min(m, n);
      let ni = Math.min(m + 1, n);
      let s = new Float64Array(ni);
      let U = new Matrix(m, nu);
      let V = new Matrix(n, n);
      let e = new Float64Array(n);
      let work = new Float64Array(m);
      let si = new Float64Array(ni);

      for (let i = 0; i < ni; i++) si[i] = i;

      let nct = Math.min(m - 1, n);
      let nrt = Math.max(0, Math.min(n - 2, m));
      let mrc = Math.max(nct, nrt);

      for (let k = 0; k < mrc; k++) {
        if (k < nct) {
          s[k] = 0;

          for (let i = k; i < m; i++) {
            s[k] = hypotenuse(s[k], a.get(i, k));
          }

          if (s[k] !== 0) {
            if (a.get(k, k) < 0) {
              s[k] = -s[k];
            }

            for (let i = k; i < m; i++) {
              a.set(i, k, a.get(i, k) / s[k]);
            }

            a.set(k, k, a.get(k, k) + 1);
          }

          s[k] = -s[k];
        }

        for (let j = k + 1; j < n; j++) {
          if (k < nct && s[k] !== 0) {
            let t = 0;

            for (let i = k; i < m; i++) {
              t += a.get(i, k) * a.get(i, j);
            }

            t = -t / a.get(k, k);

            for (let i = k; i < m; i++) {
              a.set(i, j, a.get(i, j) + t * a.get(i, k));
            }
          }

          e[j] = a.get(k, j);
        }

        if (wantu && k < nct) {
          for (let i = k; i < m; i++) {
            U.set(i, k, a.get(i, k));
          }
        }

        if (k < nrt) {
          e[k] = 0;

          for (let i = k + 1; i < n; i++) {
            e[k] = hypotenuse(e[k], e[i]);
          }

          if (e[k] !== 0) {
            if (e[k + 1] < 0) {
              e[k] = 0 - e[k];
            }

            for (let i = k + 1; i < n; i++) {
              e[i] /= e[k];
            }

            e[k + 1] += 1;
          }

          e[k] = -e[k];

          if (k + 1 < m && e[k] !== 0) {
            for (let i = k + 1; i < m; i++) {
              work[i] = 0;
            }

            for (let i = k + 1; i < m; i++) {
              for (let j = k + 1; j < n; j++) {
                work[i] += e[j] * a.get(i, j);
              }
            }

            for (let j = k + 1; j < n; j++) {
              let t = -e[j] / e[k + 1];

              for (let i = k + 1; i < m; i++) {
                a.set(i, j, a.get(i, j) + t * work[i]);
              }
            }
          }

          if (wantv) {
            for (let i = k + 1; i < n; i++) {
              V.set(i, k, e[i]);
            }
          }
        }
      }

      let p = Math.min(n, m + 1);

      if (nct < n) {
        s[nct] = a.get(nct, nct);
      }

      if (m < p) {
        s[p - 1] = 0;
      }

      if (nrt + 1 < p) {
        e[nrt] = a.get(nrt, p - 1);
      }

      e[p - 1] = 0;

      if (wantu) {
        for (let j = nct; j < nu; j++) {
          for (let i = 0; i < m; i++) {
            U.set(i, j, 0);
          }

          U.set(j, j, 1);
        }

        for (let k = nct - 1; k >= 0; k--) {
          if (s[k] !== 0) {
            for (let j = k + 1; j < nu; j++) {
              let t = 0;

              for (let i = k; i < m; i++) {
                t += U.get(i, k) * U.get(i, j);
              }

              t = -t / U.get(k, k);

              for (let i = k; i < m; i++) {
                U.set(i, j, U.get(i, j) + t * U.get(i, k));
              }
            }

            for (let i = k; i < m; i++) {
              U.set(i, k, -U.get(i, k));
            }

            U.set(k, k, 1 + U.get(k, k));

            for (let i = 0; i < k - 1; i++) {
              U.set(i, k, 0);
            }
          } else {
            for (let i = 0; i < m; i++) {
              U.set(i, k, 0);
            }

            U.set(k, k, 1);
          }
        }
      }

      if (wantv) {
        for (let k = n - 1; k >= 0; k--) {
          if (k < nrt && e[k] !== 0) {
            for (let j = k + 1; j < n; j++) {
              let t = 0;

              for (let i = k + 1; i < n; i++) {
                t += V.get(i, k) * V.get(i, j);
              }

              t = -t / V.get(k + 1, k);

              for (let i = k + 1; i < n; i++) {
                V.set(i, j, V.get(i, j) + t * V.get(i, k));
              }
            }
          }

          for (let i = 0; i < n; i++) {
            V.set(i, k, 0);
          }

          V.set(k, k, 1);
        }
      }

      let pp = p - 1;
      let eps = Number.EPSILON;

      while (p > 0) {
        let k, kase;

        for (k = p - 2; k >= -1; k--) {
          if (k === -1) {
            break;
          }

          const alpha = Number.MIN_VALUE + eps * Math.abs(s[k] + Math.abs(s[k + 1]));

          if (Math.abs(e[k]) <= alpha || Number.isNaN(e[k])) {
            e[k] = 0;
            break;
          }
        }

        if (k === p - 2) {
          kase = 4;
        } else {
          let ks;

          for (ks = p - 1; ks >= k; ks--) {
            if (ks === k) {
              break;
            }

            let t = (ks !== p ? Math.abs(e[ks]) : 0) + (ks !== k + 1 ? Math.abs(e[ks - 1]) : 0);

            if (Math.abs(s[ks]) <= eps * t) {
              s[ks] = 0;
              break;
            }
          }

          if (ks === k) {
            kase = 3;
          } else if (ks === p - 1) {
            kase = 1;
          } else {
            kase = 2;
            k = ks;
          }
        }

        k++;

        switch (kase) {
          case 1:
            {
              let f = e[p - 2];
              e[p - 2] = 0;

              for (let j = p - 2; j >= k; j--) {
                let t = hypotenuse(s[j], f);
                let cs = s[j] / t;
                let sn = f / t;
                s[j] = t;

                if (j !== k) {
                  f = -sn * e[j - 1];
                  e[j - 1] = cs * e[j - 1];
                }

                if (wantv) {
                  for (let i = 0; i < n; i++) {
                    t = cs * V.get(i, j) + sn * V.get(i, p - 1);
                    V.set(i, p - 1, -sn * V.get(i, j) + cs * V.get(i, p - 1));
                    V.set(i, j, t);
                  }
                }
              }

              break;
            }

          case 2:
            {
              let f = e[k - 1];
              e[k - 1] = 0;

              for (let j = k; j < p; j++) {
                let t = hypotenuse(s[j], f);
                let cs = s[j] / t;
                let sn = f / t;
                s[j] = t;
                f = -sn * e[j];
                e[j] = cs * e[j];

                if (wantu) {
                  for (let i = 0; i < m; i++) {
                    t = cs * U.get(i, j) + sn * U.get(i, k - 1);
                    U.set(i, k - 1, -sn * U.get(i, j) + cs * U.get(i, k - 1));
                    U.set(i, j, t);
                  }
                }
              }

              break;
            }

          case 3:
            {
              const scale = Math.max(Math.abs(s[p - 1]), Math.abs(s[p - 2]), Math.abs(e[p - 2]), Math.abs(s[k]), Math.abs(e[k]));
              const sp = s[p - 1] / scale;
              const spm1 = s[p - 2] / scale;
              const epm1 = e[p - 2] / scale;
              const sk = s[k] / scale;
              const ek = e[k] / scale;
              const b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2;
              const c = sp * epm1 * (sp * epm1);
              let shift = 0;

              if (b !== 0 || c !== 0) {
                if (b < 0) {
                  shift = 0 - Math.sqrt(b * b + c);
                } else {
                  shift = Math.sqrt(b * b + c);
                }

                shift = c / (b + shift);
              }

              let f = (sk + sp) * (sk - sp) + shift;
              let g = sk * ek;

              for (let j = k; j < p - 1; j++) {
                let t = hypotenuse(f, g);
                if (t === 0) t = Number.MIN_VALUE;
                let cs = f / t;
                let sn = g / t;

                if (j !== k) {
                  e[j - 1] = t;
                }

                f = cs * s[j] + sn * e[j];
                e[j] = cs * e[j] - sn * s[j];
                g = sn * s[j + 1];
                s[j + 1] = cs * s[j + 1];

                if (wantv) {
                  for (let i = 0; i < n; i++) {
                    t = cs * V.get(i, j) + sn * V.get(i, j + 1);
                    V.set(i, j + 1, -sn * V.get(i, j) + cs * V.get(i, j + 1));
                    V.set(i, j, t);
                  }
                }

                t = hypotenuse(f, g);
                if (t === 0) t = Number.MIN_VALUE;
                cs = f / t;
                sn = g / t;
                s[j] = t;
                f = cs * e[j] + sn * s[j + 1];
                s[j + 1] = -sn * e[j] + cs * s[j + 1];
                g = sn * e[j + 1];
                e[j + 1] = cs * e[j + 1];

                if (wantu && j < m - 1) {
                  for (let i = 0; i < m; i++) {
                    t = cs * U.get(i, j) + sn * U.get(i, j + 1);
                    U.set(i, j + 1, -sn * U.get(i, j) + cs * U.get(i, j + 1));
                    U.set(i, j, t);
                  }
                }
              }

              e[p - 2] = f;
              break;
            }

          case 4:
            {
              if (s[k] <= 0) {
                s[k] = s[k] < 0 ? -s[k] : 0;

                if (wantv) {
                  for (let i = 0; i <= pp; i++) {
                    V.set(i, k, -V.get(i, k));
                  }
                }
              }

              while (k < pp) {
                if (s[k] >= s[k + 1]) {
                  break;
                }

                let t = s[k];
                s[k] = s[k + 1];
                s[k + 1] = t;

                if (wantv && k < n - 1) {
                  for (let i = 0; i < n; i++) {
                    t = V.get(i, k + 1);
                    V.set(i, k + 1, V.get(i, k));
                    V.set(i, k, t);
                  }
                }

                if (wantu && k < m - 1) {
                  for (let i = 0; i < m; i++) {
                    t = U.get(i, k + 1);
                    U.set(i, k + 1, U.get(i, k));
                    U.set(i, k, t);
                  }
                }

                k++;
              }
              p--;
              break;
            }
          // no default
        }
      }

      if (swapped) {
        let tmp = V;
        V = U;
        U = tmp;
      }

      this.m = m;
      this.n = n;
      this.s = s;
      this.U = U;
      this.V = V;
    }

    solve(value) {
      let Y = value;
      let e = this.threshold;
      let scols = this.s.length;
      let Ls = Matrix.zeros(scols, scols);

      for (let i = 0; i < scols; i++) {
        if (Math.abs(this.s[i]) <= e) {
          Ls.set(i, i, 0);
        } else {
          Ls.set(i, i, 1 / this.s[i]);
        }
      }

      let U = this.U;
      let V = this.rightSingularVectors;
      let VL = V.mmul(Ls);
      let vrows = V.rows;
      let urows = U.rows;
      let VLU = Matrix.zeros(vrows, urows);

      for (let i = 0; i < vrows; i++) {
        for (let j = 0; j < urows; j++) {
          let sum = 0;

          for (let k = 0; k < scols; k++) {
            sum += VL.get(i, k) * U.get(j, k);
          }

          VLU.set(i, j, sum);
        }
      }

      return VLU.mmul(Y);
    }

    solveForDiagonal(value) {
      return this.solve(Matrix.diag(value));
    }

    inverse() {
      let V = this.V;
      let e = this.threshold;
      let vrows = V.rows;
      let vcols = V.columns;
      let X = new Matrix(vrows, this.s.length);

      for (let i = 0; i < vrows; i++) {
        for (let j = 0; j < vcols; j++) {
          if (Math.abs(this.s[j]) > e) {
            X.set(i, j, V.get(i, j) / this.s[j]);
          }
        }
      }

      let U = this.U;
      let urows = U.rows;
      let ucols = U.columns;
      let Y = new Matrix(vrows, urows);

      for (let i = 0; i < vrows; i++) {
        for (let j = 0; j < urows; j++) {
          let sum = 0;

          for (let k = 0; k < ucols; k++) {
            sum += X.get(i, k) * U.get(j, k);
          }

          Y.set(i, j, sum);
        }
      }

      return Y;
    }

    get condition() {
      return this.s[0] / this.s[Math.min(this.m, this.n) - 1];
    }

    get norm2() {
      return this.s[0];
    }

    get rank() {
      let tol = Math.max(this.m, this.n) * this.s[0] * Number.EPSILON;
      let r = 0;
      let s = this.s;

      for (let i = 0, ii = s.length; i < ii; i++) {
        if (s[i] > tol) {
          r++;
        }
      }

      return r;
    }

    get diagonal() {
      return Array.from(this.s);
    }

    get threshold() {
      return Number.EPSILON / 2 * Math.max(this.m, this.n) * this.s[0];
    }

    get leftSingularVectors() {
      return this.U;
    }

    get rightSingularVectors() {
      return this.V;
    }

    get diagonalMatrix() {
      return Matrix.diag(this.s);
    }

  }

  function inverse(matrix) {
    let useSVD = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    matrix = WrapperMatrix2D.checkMatrix(matrix);

    if (useSVD) {
      return new SingularValueDecomposition(matrix).inverse();
    } else {
      return solve(matrix, Matrix.eye(matrix.rows));
    }
  }
  function solve(leftHandSide, rightHandSide) {
    let useSVD = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
    leftHandSide = WrapperMatrix2D.checkMatrix(leftHandSide);
    rightHandSide = WrapperMatrix2D.checkMatrix(rightHandSide);

    if (useSVD) {
      return new SingularValueDecomposition(leftHandSide).solve(rightHandSide);
    } else {
      return leftHandSide.isSquare() ? new LuDecomposition(leftHandSide).solve(rightHandSide) : new QrDecomposition(leftHandSide).solve(rightHandSide);
    }
  }

  class nipals {
    constructor(X) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
      X = WrapperMatrix2D.checkMatrix(X);
      let {
        Y
      } = options;
      const {
        scaleScores = false,
        maxIterations = 1000,
        terminationCriteria = 1e-10
      } = options;
      let u;

      if (Y) {
        if (Array.isArray(Y) && typeof Y[0] === 'number') {
          Y = Matrix.columnVector(Y);
        } else {
          Y = WrapperMatrix2D.checkMatrix(Y);
        }

        if (!Y.isColumnVector() || Y.rows !== X.rows) {
          throw new Error('Y must be a column vector of length X.rows');
        }

        u = Y;
      } else {
        u = X.getColumnVector(0);
      }

      let diff = 1;
      let t, q, w, tOld;

      for (let counter = 0; counter < maxIterations && diff > terminationCriteria; counter++) {
        w = X.transpose().mmul(u).div(u.transpose().mmul(u).get(0, 0));
        w = w.div(w.norm());
        t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0));

        if (counter > 0) {
          diff = t.clone().sub(tOld).pow(2).sum();
        }

        tOld = t.clone();

        if (Y) {
          q = Y.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
          q = q.div(q.norm());
          u = Y.mmul(q).div(q.transpose().mmul(q).get(0, 0));
        } else {
          u = t;
        }
      }

      if (Y) {
        let p = X.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
        p = p.div(p.norm());
        let xResidual = X.clone().sub(t.clone().mmul(p.transpose()));
        let residual = u.transpose().mmul(t).div(t.transpose().mmul(t).get(0, 0));
        let yResidual = Y.clone().sub(t.clone().mulS(residual.get(0, 0)).mmul(q.transpose()));
        this.t = t;
        this.p = p.transpose();
        this.w = w.transpose();
        this.q = q;
        this.u = u;
        this.s = t.transpose().mmul(t);
        this.xResidual = xResidual;
        this.yResidual = yResidual;
        this.betas = residual;
      } else {
        this.w = w.transpose();
        this.s = t.transpose().mmul(t).sqrt();

        if (scaleScores) {
          this.t = t.clone().div(this.s.get(0, 0));
        } else {
          this.t = t;
        }

        this.xResidual = X.sub(t.mmul(w.transpose()));
      }
    }

  }

  /**
   * @private
   * Function that given vector, returns its norm
   * @param {Vector} X
   * @return {number} Norm of the vector
   */

  function norm(X) {
    return Math.sqrt(X.clone().apply(pow2array).sum());
  }
  /**
   * @private
   * Function that pow 2 each element of a Matrix or a Vector,
   * used in the apply method of the Matrix object
   * @param {number} i - index i.
   * @param {number} j - index j.
   * @return {Matrix} The Matrix object modified at the index i, j.
   * */

  function pow2array(i, j) {
    this.set(i, j, this.get(i, j) ** 2);
  }
  /**
   * @private
   * Function that initialize an array of matrices.
   * @param {Array} array
   * @param {boolean} isMatrix
   * @return {Array} array with the matrices initialized.
   */

  function initializeMatrices(array, isMatrix) {
    if (isMatrix) {
      for (let i = 0; i < array.length; ++i) {
        for (let j = 0; j < array[i].length; ++j) {
          let elem = array[i][j];
          array[i][j] = elem !== null ? new Matrix(array[i][j]) : undefined;
        }
      }
    } else {
      for (let i = 0; i < array.length; ++i) {
        array[i] = new Matrix(array[i]);
      }
    }

    return array;
  } // /**
  function summaryMetadata(classVector) {
    let nObs = classVector.length;
    let type = typeof classVector[0];
    let counts = {};

    switch (type) {
      case 'string':
        counts = {};
        classVector.forEach(x => counts[x] = (counts[x] || 0) + 1);
        break;

      case 'number':
        classVector = classVector.map(x => x.toString());
        counts = {};
        classVector.forEach(x => counts[x] = (counts[x] || 0) + 1);
        break;
    }

    let groupIDs = Object.keys(counts);
    let nClass = groupIDs.length;
    let classFactor = classVector.map(x => groupIDs.indexOf(x));
    let classMatrix = Matrix.from1DArray(nObs, 1, classFactor);
    return {
      groupIDs,
      nClass,
      classVector,
      classFactor,
      classMatrix
    };
  }

  /**
   * @class PLS
   */

  class PLS {
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
        let {
          tolerance = 1e-5,
          scale = true
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

      this.meanX = trainingSet.mean('column');
      this.stdDevX = trainingSet.standardDeviation('column', {
        mean: this.meanX,
        unbiased: true
      });
      this.meanY = trainingValues.mean('column');
      this.stdDevY = trainingValues.standardDeviation('column', {
        mean: this.meanY,
        unbiased: true
      });

      if (this.scale) {
        trainingSet = trainingSet.clone().subRowVector(this.meanX).divRowVector(this.stdDevX);
        trainingValues = trainingValues.clone().subRowVector(this.meanY).divRowVector(this.stdDevY);
      }

      if (this.latentVectors === undefined) {
        this.latentVectors = Math.min(trainingSet.rows - 1, trainingSet.columns);
      }

      let rx = trainingSet.rows;
      let cx = trainingSet.columns;
      let ry = trainingValues.rows;
      let cy = trainingValues.columns;
      let ssqXcal = trainingSet.clone().mul(trainingSet).sum(); // for the r²

      let sumOfSquaresY = trainingValues.clone().mul(trainingValues).sum();
      let tolerance = this.tolerance;
      let n = this.latentVectors;
      let T = Matrix.zeros(rx, n);
      let P = Matrix.zeros(cx, n);
      let U = Matrix.zeros(ry, n);
      let Q = Matrix.zeros(cy, n);
      let B = Matrix.zeros(n, n);
      let W = P.clone();
      let k = 0;

      while (norm(trainingValues) > tolerance && k < n) {
        let transposeX = trainingSet.transpose();
        let transposeY = trainingValues.transpose();
        let tIndex = maxSumColIndex(trainingSet.clone().mul(trainingSet));
        let uIndex = maxSumColIndex(trainingValues.clone().mul(trainingValues));
        let t1 = trainingSet.getColumnVector(tIndex);
        let u = trainingValues.getColumnVector(uIndex);
        var t = Matrix.zeros(rx, 1);

        while (norm(t1.clone().sub(t)) > tolerance) {
          var w = transposeX.mmul(u);
          w.div(norm(w));
          t = t1;
          t1 = trainingSet.mmul(w);
          var q = transposeY.mmul(t1);
          q.div(norm(q));
          u = trainingValues.mmul(q);
        }

        t = t1;
        let num = transposeX.mmul(t);
        let den = t.transpose().mmul(t).get(0, 0);
        var p = num.div(den);
        let pnorm = norm(p);
        p.div(pnorm);
        t.mul(pnorm);
        w.mul(pnorm);
        num = u.transpose().mmul(t);
        den = t.transpose().mmul(t).get(0, 0);
        let b = num.div(den).get(0, 0);
        trainingSet.sub(t.mmul(p.transpose()));
        trainingValues.sub(t.clone().mul(b).mmul(q.transpose()));
        T.setColumn(k, t);
        P.setColumn(k, p);
        U.setColumn(k, u);
        Q.setColumn(k, q);
        W.setColumn(k, w);
        B.set(k, k, b);
        k++;
      }

      k--;
      T = T.subMatrix(0, T.rows - 1, 0, k);
      P = P.subMatrix(0, P.rows - 1, 0, k);
      U = U.subMatrix(0, U.rows - 1, 0, k);
      Q = Q.subMatrix(0, Q.rows - 1, 0, k);
      W = W.subMatrix(0, W.rows - 1, 0, k);
      B = B.subMatrix(0, k, 0, k); // TODO: review of R2Y
      // this.R2Y = t.transpose().mmul(t).mul(q[k][0]*q[k][0]).divS(ssqYcal)[0][0];
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
      this.R2X = t.transpose().mmul(t).mmul(p.transpose().mmul(p)).div(ssqXcal).get(0, 0);
    }
    /**
     * Predicts the behavior of the given dataset.
     * @param {Matrix|Array} dataset - data to be predicted.
     * @return {Matrix} - predictions of each element of the dataset.
     */


    predict(dataset) {
      let X = Matrix.checkMatrix(dataset);

      if (this.scale) {
        X = X.subRowVector(this.meanX).divRowVector(this.stdDevX);
      }

      let Y = X.mmul(this.PBQ);
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
        scale: this.scale
      };
    }
    /**
     * Load a PLS model from a JSON Object
     * @param {object} model
     * @return {PLS} - PLS object from the given model
     */


    static load(model) {
      if (model.name !== 'PLS') {
        throw new RangeError("Invalid model: ".concat(model.name));
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
    return Matrix.rowVector(data.sum('column')).maxIndex()[0];
  }

  /**
   * @class KOPLS
   */

  class KOPLS {
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
      trainingValues = Matrix.checkMatrix(trainingValues); // to save and compute kernel with the prediction dataset.

      this.trainingSet = trainingSet.clone();
      let kernelX = this.kernel.compute(trainingSet);
      let Identity = Matrix.eye(kernelX.rows, kernelX.rows, 1);
      let temp = kernelX;
      kernelX = new Array(this.orthogonalComp + 1);

      for (let i = 0; i < this.orthogonalComp + 1; i++) {
        kernelX[i] = new Array(this.orthogonalComp + 1);
      }

      kernelX[0][0] = temp;
      let result = new SingularValueDecomposition(trainingValues.transpose().mmul(kernelX[0][0]).mmul(trainingValues), {
        computeLeftSingularVectors: true,
        computeRightSingularVectors: false
      });
      let YLoadingMat = result.leftSingularVectors;
      let Sigma = result.diagonalMatrix;
      YLoadingMat = YLoadingMat.subMatrix(0, YLoadingMat.rows - 1, 0, this.predictiveComp - 1);
      Sigma = Sigma.subMatrix(0, this.predictiveComp - 1, 0, this.predictiveComp - 1);
      let YScoreMat = trainingValues.mmul(YLoadingMat);
      let predScoreMat = new Array(this.orthogonalComp + 1);
      let TURegressionCoeff = new Array(this.orthogonalComp + 1);
      let YOrthScoreMat = new Array(this.orthogonalComp);
      let YOrthLoadingVec = new Array(this.orthogonalComp);
      let YOrthEigen = new Array(this.orthogonalComp);
      let YOrthScoreNorm = new Array(this.orthogonalComp);
      let SigmaPow = Matrix.pow(Sigma, -0.5); // to avoid errors, check infinity

      SigmaPow.apply(function (i, j) {
        if (this.get(i, j) === Infinity) {
          this.set(i, j, 0);
        }
      });

      for (let i = 0; i < this.orthogonalComp; ++i) {
        predScoreMat[i] = kernelX[0][i].transpose().mmul(YScoreMat).mmul(SigmaPow);
        let TpiPrime = predScoreMat[i].transpose();
        TURegressionCoeff[i] = inverse(TpiPrime.mmul(predScoreMat[i])).mmul(TpiPrime).mmul(YScoreMat);
        result = new SingularValueDecomposition(TpiPrime.mmul(Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime))).mmul(predScoreMat[i]), {
          computeLeftSingularVectors: true,
          computeRightSingularVectors: false
        });
        let CoTemp = result.leftSingularVectors;
        let SoTemp = result.diagonalMatrix;
        YOrthLoadingVec[i] = CoTemp.subMatrix(0, CoTemp.rows - 1, 0, 0);
        YOrthEigen[i] = SoTemp.get(0, 0);
        YOrthScoreMat[i] = Matrix.sub(kernelX[i][i], predScoreMat[i].mmul(TpiPrime)).mmul(predScoreMat[i]).mmul(YOrthLoadingVec[i]).mul(Math.pow(YOrthEigen[i], -0.5));
        let toiPrime = YOrthScoreMat[i].transpose();
        YOrthScoreNorm[i] = Matrix.sqrt(toiPrime.mmul(YOrthScoreMat[i]));
        YOrthScoreMat[i] = YOrthScoreMat[i].divRowVector(YOrthScoreNorm[i]);
        let ITo = Matrix.sub(Identity, YOrthScoreMat[i].mmul(YOrthScoreMat[i].transpose()));
        kernelX[0][i + 1] = kernelX[0][i].mmul(ITo);
        kernelX[i + 1][i + 1] = ITo.mmul(kernelX[i][i]).mmul(ITo);
      }

      let lastScoreMat = predScoreMat[this.orthogonalComp] = kernelX[0][this.orthogonalComp].transpose().mmul(YScoreMat).mmul(SigmaPow);
      let lastTpPrime = lastScoreMat.transpose();
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
      let KTestTrain = this.kernel.compute(toPredict, this.trainingSet);
      let temp = KTestTrain;
      KTestTrain = new Array(this.orthogonalComp + 1);

      for (let i = 0; i < this.orthogonalComp + 1; i++) {
        KTestTrain[i] = new Array(this.orthogonalComp + 1);
      }

      KTestTrain[0][0] = temp;
      let YOrthScoreVector = new Array(this.orthogonalComp);
      let predScoreMat = new Array(this.orthogonalComp);
      let i;

      for (i = 0; i < this.orthogonalComp; ++i) {
        predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);
        YOrthScoreVector[i] = Matrix.sub(KTestTrain[i][i], predScoreMat[i].mmul(this.predScoreMat[i].transpose())).mmul(this.predScoreMat[i]).mmul(this.YOrthLoadingVec[i]).mul(Math.pow(this.YOrthEigen[i], -0.5));
        YOrthScoreVector[i] = YOrthScoreVector[i].divRowVector(this.toNorm[i]);
        let scoreMatPrime = this.YOrthScoreMat[i].transpose();
        KTestTrain[i + 1][0] = Matrix.sub(KTestTrain[i][0], YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[0][i].transpose()));
        let p1 = Matrix.sub(KTestTrain[i][0], KTestTrain[i][i].mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime));
        let p2 = YOrthScoreVector[i].mmul(scoreMatPrime).mmul(this.kernelX[i][i]);
        let p3 = p2.mmul(this.YOrthScoreMat[i]).mmul(scoreMatPrime);
        KTestTrain[i + 1][i + 1] = p1.sub(p2).add(p3);
      }

      predScoreMat[i] = KTestTrain[i][0].mmul(this.YScoreMat).mmul(this.SigmaPow);
      let prediction = predScoreMat[i].mmul(this.TURegressionCoeff[i]).mmul(this.YLoadingMat.transpose());
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
        throw new RangeError("Invalid model: ".concat(model.name));
      }

      if (!kernel) {
        throw new RangeError('You must provide a kernel for the model!');
      }

      model.kernel = kernel;
      return new KOPLS(true, model);
    }

  }

  /**
   *  Constructs a confusion matrix
   * @class ConfusionMatrix
   * @example
   * const CM = new ConfusionMatrix([[13, 2], [10, 5]], ['cat', 'dog'])
   * @param {Array<Array<number>>} matrix - The confusion matrix, a 2D Array. Rows represent the actual label and columns
   *     the predicted label.
   * @param {Array<any>} labels - Labels of the confusion matrix, a 1D Array
   */
  class ConfusionMatrix {
    constructor(matrix, labels) {
      if (matrix.length !== matrix[0].length) {
        throw new Error('Confusion matrix must be square');
      }

      if (labels.length !== matrix.length) {
        throw new Error('Confusion matrix and labels should have the same length');
      }

      this.labels = labels;
      this.matrix = matrix;
    }
    /**
     * Construct confusion matrix from the predicted and actual labels (classes). Be sure to provide the arguments in
     * the correct order!
     * @param {Array<any>} actual  - The predicted labels of the classification
     * @param {Array<any>} predicted     - The actual labels of the classification. Has to be of same length as
     *     predicted.
     * @param {object} [options] - Additional options
     * @param {Array<any>} [options.labels] - The list of labels that should be used. If not provided the distinct set
     *     of labels present in predicted and actual is used. Labels are compared using the strict equality operator
     *     '==='
     * @return {ConfusionMatrix} - Confusion matrix
     */


    static fromLabels(actual, predicted) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

      if (predicted.length !== actual.length) {
        throw new Error('predicted and actual must have the same length');
      }

      let distinctLabels;

      if (options.labels) {
        distinctLabels = new Set(options.labels);
      } else {
        distinctLabels = new Set([...actual, ...predicted]);
      }

      distinctLabels = Array.from(distinctLabels);

      if (options.sort) {
        distinctLabels.sort(options.sort);
      } // Create confusion matrix and fill with 0's


      const matrix = Array.from({
        length: distinctLabels.length
      });

      for (let i = 0; i < matrix.length; i++) {
        matrix[i] = new Array(matrix.length);
        matrix[i].fill(0);
      }

      for (let i = 0; i < predicted.length; i++) {
        const actualIdx = distinctLabels.indexOf(actual[i]);
        const predictedIdx = distinctLabels.indexOf(predicted[i]);

        if (actualIdx >= 0 && predictedIdx >= 0) {
          matrix[actualIdx][predictedIdx]++;
        }
      }

      return new ConfusionMatrix(matrix, distinctLabels);
    }
    /**
     * Get the confusion matrix
     * @return {Array<Array<number> >}
     */


    getMatrix() {
      return this.matrix;
    }

    getLabels() {
      return this.labels;
    }
    /**
     * Get the total number of samples
     * @return {number}
     */


    getTotalCount() {
      let predicted = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          predicted += this.matrix[i][j];
        }
      }

      return predicted;
    }
    /**
     * Get the total number of true predictions
     * @return {number}
     */


    getTrueCount() {
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        count += this.matrix[i][i];
      }

      return count;
    }
    /**
     * Get the total number of false predictions.
     * @return {number}
     */


    getFalseCount() {
      return this.getTotalCount() - this.getTrueCount();
    }
    /**
     * Get the number of true positive predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTruePositiveCount(label) {
      const index = this.getIndex(label);
      return this.matrix[index][index];
    }
    /**
     * Get the number of true negative predictions
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTrueNegativeCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          if (i !== index && j !== index) {
            count += this.matrix[i][j];
          }
        }
      }

      return count;
    }
    /**
     * Get the number of false positive predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalsePositiveCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        if (i !== index) {
          count += this.matrix[i][index];
        }
      }

      return count;
    }
    /**
     * Get the number of false negative predictions.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseNegativeCount(label) {
      const index = this.getIndex(label);
      let count = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        if (i !== index) {
          count += this.matrix[index][i];
        }
      }

      return count;
    }
    /**
     * Get the number of real positive samples.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getPositiveCount(label) {
      return this.getTruePositiveCount(label) + this.getFalseNegativeCount(label);
    }
    /**
     * Get the number of real negative samples.
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getNegativeCount(label) {
      return this.getTrueNegativeCount(label) + this.getFalsePositiveCount(label);
    }
    /**
     * Get the index in the confusion matrix that corresponds to the given label
     * @param {any} label - The label to search for
     * @throws if the label is not found
     * @return {number}
     */


    getIndex(label) {
      const index = this.labels.indexOf(label);
      if (index === -1) throw new Error('The label does not exist');
      return index;
    }
    /**
     * Get the true positive rate a.k.a. sensitivity. Computes the ratio between the number of true positive predictions and the total number of positive samples.
     * {@link https://en.wikipedia.org/wiki/Sensitivity_and_specificity}
     * @param {any} label - The label that should be considered "positive"
     * @return {number} - The true positive rate [0-1]
     */


    getTruePositiveRate(label) {
      return this.getTruePositiveCount(label) / this.getPositiveCount(label);
    }
    /**
     * Get the true negative rate a.k.a. specificity. Computes the ration between the number of true negative predictions and the total number of negative samples.
     * {@link https://en.wikipedia.org/wiki/Sensitivity_and_specificity}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getTrueNegativeRate(label) {
      return this.getTrueNegativeCount(label) / this.getNegativeCount(label);
    }
    /**
     * Get the positive predictive value a.k.a. precision. Computes TP / (TP + FP)
     * {@link https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getPositivePredictiveValue(label) {
      const TP = this.getTruePositiveCount(label);
      return TP / (TP + this.getFalsePositiveCount(label));
    }
    /**
     * Negative predictive value
     * {@link https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getNegativePredictiveValue(label) {
      const TN = this.getTrueNegativeCount(label);
      return TN / (TN + this.getFalseNegativeCount(label));
    }
    /**
     * False negative rate a.k.a. miss rate.
     * {@link https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseNegativeRate(label) {
      return 1 - this.getTruePositiveRate(label);
    }
    /**
     * False positive rate a.k.a. fall-out rate.
     * {@link https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalsePositiveRate(label) {
      return 1 - this.getTrueNegativeRate(label);
    }
    /**
     * False discovery rate (FDR)
     * {@link https://en.wikipedia.org/wiki/False_discovery_rate}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseDiscoveryRate(label) {
      const FP = this.getFalsePositiveCount(label);
      return FP / (FP + this.getTruePositiveCount(label));
    }
    /**
     * False omission rate (FOR)
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getFalseOmissionRate(label) {
      const FN = this.getFalseNegativeCount(label);
      return FN / (FN + this.getTruePositiveCount(label));
    }
    /**
     * F1 score
     * {@link https://en.wikipedia.org/wiki/F1_score}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getF1Score(label) {
      const TP = this.getTruePositiveCount(label);
      return 2 * TP / (2 * TP + this.getFalsePositiveCount(label) + this.getFalseNegativeCount(label));
    }
    /**
     * Matthews correlation coefficient (MCC)
     * {@link https://en.wikipedia.org/wiki/Matthews_correlation_coefficient}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getMatthewsCorrelationCoefficient(label) {
      const TP = this.getTruePositiveCount(label);
      const TN = this.getTrueNegativeCount(label);
      const FP = this.getFalsePositiveCount(label);
      const FN = this.getFalseNegativeCount(label);
      return (TP * TN - FP * FN) / Math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
    }
    /**
     * Informedness
     * {@link https://en.wikipedia.org/wiki/Youden%27s_J_statistic}
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getInformedness(label) {
      return this.getTruePositiveRate(label) + this.getTrueNegativeRate(label) - 1;
    }
    /**
     * Markedness
     * @param {any} label - The label that should be considered "positive"
     * @return {number}
     */


    getMarkedness(label) {
      return this.getPositivePredictiveValue(label) + this.getNegativePredictiveValue(label) - 1;
    }
    /**
     * Get the confusion table.
     * @param {any} label - The label that should be considered "positive"
     * @return {Array<Array<number> >} - The 2x2 confusion table. [[TP, FN], [FP, TN]]
     */


    getConfusionTable(label) {
      return [[this.getTruePositiveCount(label), this.getFalseNegativeCount(label)], [this.getFalsePositiveCount(label), this.getTrueNegativeCount(label)]];
    }
    /**
     * Get total accuracy.
     * @return {number} - The ratio between the number of true predictions and total number of classifications ([0-1])
     */


    getAccuracy() {
      let correct = 0;
      let incorrect = 0;

      for (let i = 0; i < this.matrix.length; i++) {
        for (let j = 0; j < this.matrix.length; j++) {
          if (i === j) correct += this.matrix[i][j];else incorrect += this.matrix[i][j];
        }
      }

      return correct / (correct + incorrect);
    }
    /**
     * Returns the element in the confusion matrix that corresponds to the given actual and predicted labels.
     * @param {any} actual - The true label
     * @param {any} predicted - The predicted label
     * @return {number} - The element in the confusion matrix
     */


    getCount(actual, predicted) {
      const actualIndex = this.getIndex(actual);
      const predictedIndex = this.getIndex(predicted);
      return this.matrix[actualIndex][predictedIndex];
    }
    /**
     * Compute the general prediction accuracy
     * @deprecated Use getAccuracy
     * @return {number} - The prediction accuracy ([0-1]
     */


    get accuracy() {
      return this.getAccuracy();
    }
    /**
     * Compute the number of predicted observations
     * @deprecated Use getTotalCount
     * @return {number}
     */


    get total() {
      return this.getTotalCount();
    }

  }

  /**
   * get folds indexes
   * @param {Array} features
   * @param {Number} k - number of folds, a
   */
  function getFolds(features) {
    let k = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 5;
    let N = features.length;
    let allIdx = new Array(N);

    for (let i = 0; i < N; i++) {
      allIdx[i] = i;
    }

    let l = Math.floor(N / k); // create random k-folds

    let current = [];
    let folds = [];

    while (allIdx.length) {
      let randi = Math.floor(Math.random() * allIdx.length);
      current.push(allIdx[randi]);
      allIdx.splice(randi, 1);

      if (current.length === l) {
        folds.push(current);
        current = [];
      }
    } // we push the remaining to the last fold so that the total length is
    // preserved. Otherwise the Q2 will fail.


    if (current.length) current.forEach(e => folds[k - 1].push(e));
    folds = folds.slice(0, k);
    let foldsIndex = folds.map((x, idx) => ({
      testIndex: x,
      trainIndex: [].concat(...folds.filter((el, idx2) => idx2 !== idx))
    }));
    return foldsIndex;
  }

  let Utils = {};

  Utils.norm = function norm(X) {
    return Math.sqrt(X.clone().apply(pow2array).sum());
  };
  /**
   * OPLS loop
   * @param {Array} x a matrix with features
   * @param {Array} y an array of labels (dependent variable)
   * @param {Object} options an object with options
   * @return {Object} an object with model (filteredX: err,
      loadingsXOrtho: pOrtho,
      scoresXOrtho: tOrtho,
      weightsXOrtho: wOrtho,
      weightsPred: w,
      loadingsXpred: p,
      scoresXpred: t,
      loadingsY:)
   */


  function oplsNIPALS(x, y) {
    let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
    const {
      numberOSC = 100
    } = options;
    let X = Matrix.checkMatrix(x.clone());
    let Y = Matrix.checkMatrix(y.clone());
    let u = Y.getColumnVector(0);
    let diff = 1;
    let t, c, w, uNew;

    for (let i = 0; i < numberOSC && diff > 1e-10; i++) {
      w = u.transpose().mmul(X).div(u.transpose().mmul(u).get(0, 0));
      w = w.transpose().div(norm(w));
      t = X.mmul(w).div(w.transpose().mmul(w).get(0, 0)); // t_h paso 3
      // calc loading

      c = t.transpose().mmul(Y).div(t.transpose().mmul(t).get(0, 0)); // calc new u and compare with one in previus iteration (stop criterion)

      uNew = Y.mmul(c.transpose());
      uNew = uNew.div(c.transpose().mmul(c).get(0, 0));

      if (i > 0) {
        diff = uNew.clone().sub(u).pow(2).sum() / uNew.clone().pow(2).sum();
      }

      u = uNew.clone();
    } // calc loadings


    let p = t.transpose().mmul(X).div(t.transpose().mmul(t).get(0, 0));
    let wOrtho = p.clone().sub(w.transpose().mmul(p.transpose()).div(w.transpose().mmul(w).get(0, 0)).mmul(w.transpose()));
    wOrtho.div(Utils.norm(wOrtho)); // orthogonal scores

    let tOrtho = X.mmul(wOrtho.transpose()).div(wOrtho.mmul(wOrtho.transpose()).get(0, 0)); // orthogonal loadings

    let pOrtho = tOrtho.transpose().mmul(X).div(tOrtho.transpose().mmul(tOrtho).get(0, 0)); // filtered data

    let err = X.clone().sub(tOrtho.mmul(pOrtho));
    return {
      filteredX: err,
      weightsXOrtho: wOrtho,
      loadingsXOrtho: pOrtho,
      scoresXOrtho: tOrtho,
      weightsXPred: w,
      loadingsXpred: p,
      scoresXpred: t,
      loadingsY: c
    };
  }

  /**
   * Get total sum of square
   * @param {Array} x an array
   * @return {Number} - the sum of the squares
   */
  function tss(x) {
    return x.clone().mul(x.clone()).sum();
  }

  /**
   * Creates new OPLS (orthogonal partial latent structures) from features and labels.
   * @param {Matrix} data - matrix containing data (X).
   * @param {Array} labels - 1D Array containing metadata (Y).
   * @param {Object} [options]
   * @param {number} [options.nComp = 3] - number of latent structures computed.
   * @param {boolean} [options.center = true] - should the data be centered (subtract the mean).
   * @param {boolean} [options.scale = false] - should the data be scaled (divide by the standard deviation).
   * @param {Array} [options.cvFolds = []] - allows to provide folds as 2D array for testing purpose.
   * */

  class OPLS {
    constructor(data, labels) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

      if (data === true) {
        const opls = options;
        this.center = opls.center;
        this.scale = opls.scale;
        this.means = opls.means;
        this.stdevs = opls.stdevs;
        this.model = opls.model;
        this.tCV = opls.tCV;
        this.tOrthCV = opls.tOrthCV;
        this.mode = opls.mode;
        return;
      }

      let features = data.clone(); // set default values
      // cvFolds allows to define folds for testing purpose

      const {
        nComp = 3,
        center = true,
        scale = true,
        cvFolds = []
      } = options;
      let group;

      if (typeof labels[0] === 'number') {
        // numeric labels: OPLS regression is used
        this.mode = 'regression';
        group = Matrix.from1DArray(labels.length, 1, labels);
      } else if (typeof labels[0] === 'string') {
        // non-numeric labels: OPLS-DA is used
        this.mode = 'discriminant_analysis';
        group = labels;
        throw new Error('discriminant analysis is not yet supported');
      } // check types of features and labels


      if (features.constructor.name !== 'Matrix') {
        throw new TypeError('features must be of class Matrix');
      } // getting center and scale the features (all)


      this.center = center;

      if (this.center) {
        this.means = features.mean('column'); // console.log('training mean: ', this.means);
      } else {
        this.stdevs = null;
      }

      this.scale = scale;

      if (this.scale) {
        this.stdevs = features.standardDeviation('column'); // console.log('training sdevs: ', this.stdevs);
      } else {
        this.means = null;
      } // check and remove for features with sd = 0 TODO here
      // check opls.R line 70


      let folds;

      if (cvFolds.length > 0) {
        folds = cvFolds;
      } else {
        folds = getFolds(labels, 5);
      }

      let Q2 = [];
      this.model = [];
      this.tCV = [];
      this.tOrthCV = [];
      let oplsCV = [];
      let modelNC = []; // this code could be made more efficient by reverting the order of the loops
      // this is a legacy loop to be consistent with R code from MetaboMate package
      // this allows for having statistic (R2) from CV to decide wether to continue
      // with more latent structures

      let nc;

      for (nc = 0; nc < nComp; nc++) {
        let yHatk = new Matrix(group.rows, 1);
        let tPredk = new Matrix(group.rows, 1);
        let tOrthk = new Matrix(group.rows, 1);
        let oplsk = [];
        let f = 0;

        for (let fold of folds) {
          let trainTest = this._getTrainTest(features, group, fold);

          let testXk = trainTest.testFeatures;
          let Xk = trainTest.trainFeatures;
          let Yk = trainTest.trainLabels; // determine center and scale of training set

          let dataCenter = Xk.mean('column');
          let dataSD = Xk.standardDeviation('column'); // center and scale training set

          if (center) {
            Xk.center('column');
            Yk.center('column');
          }

          if (scale) {
            Xk.scale('column');
            Yk.scale('column');
          } // perform opls


          if (nc === 0) {
            oplsk[f] = oplsNIPALS(Xk, Yk);
          } else {
            oplsk[f] = oplsNIPALS(oplsCV[nc - 1][f].filteredX, Yk);
          } // store model for next component


          oplsCV[nc] = oplsk;
          let plsCV = new nipals(oplsk[f].filteredX, {
            Y: Yk
          }); // scaling the test dataset with respect to the train

          testXk.center('column', {
            center: dataCenter
          });
          testXk.scale('column', {
            scale: dataSD
          });
          let Eh = testXk; // removing the orthogonal components from PLS

          let scores;

          for (let idx = 0; idx < nc + 1; idx++) {
            scores = Eh.clone().mmul(oplsCV[idx][f].weightsXOrtho.transpose()); // ok

            Eh.sub(scores.clone().mmul(oplsCV[idx][f].loadingsXOrtho));
          } // prediction


          let tPred = Eh.clone().mmul(plsCV.w.transpose()); // this should be summed over ncomp (pls_prediction.R line 23)

          let yHat = tPred.clone().mmul(plsCV.betas); // ok
          // adding all prediction from all folds

          for (let i = 0; i < fold.testIndex.length; i++) {
            yHatk.setRow(fold.testIndex[i], [yHat.get(i, 0)]);
            tPredk.setRow(fold.testIndex[i], [tPred.get(i, 0)]);
            tOrthk.setRow(fold.testIndex[i], [scores.get(i, 0)]);
          }

          f++;
        } // end of loop over folds


        this.tCV.push(tPredk);
        this.tOrthCV.push(tOrthk);
        // ROC for DA is not implemented (check opls.R line 183) TODO

        if (this.mode === 'regression') {
          let tssy = tss(group.center('column').scale('column'));
          let press = tss(group.clone().sub(yHatk));
          let Q2y = 1 - press / tssy;
          Q2.push(Q2y);
        } else if (this.mode === 'discriminant_analysis') {
          throw new Error('discriminant analysis is not yet supported');
        } // calculate the R2y for the complete data


        if (nc === 0) {
          modelNC = this._predictAll(features, group);
        } else {
          modelNC = this._predictAll(modelNC.xRes, group, options = {
            scale: false,
            center: false
          });
        } // adding the predictive statistics from CV


        modelNC.Q2y = Q2; // store the model for each component

        this.model.push(modelNC); // console.warn(`OPLS iteration over # of Components: ${nc + 1}`);
      } // end of loop over nc
      // store scores from CV


      let tCV = this.tCV;
      let tOrthCV = this.tOrthCV;
      let m = this.model[nc - 1];
      let XOrth = m.XOrth;
      let FeaturesCS = features.center('column').scale('column');
      let labelsCS = group.center('column').scale('column');
      let Xres = FeaturesCS.clone().sub(XOrth);
      let plsCall = new nipals(Xres, {
        Y: labelsCS
      });
      let E = Xres.clone().sub(plsCall.t.clone().mmul(plsCall.p));
      let R2x = this.model.map(x => x.R2x);
      let R2y = this.model.map(x => x.R2y);
      this.output = {
        Q2y: Q2,
        R2x,
        R2y,
        tPred: m.plsC.t,
        pPred: m.plsC.p,
        wPred: m.plsC.w,
        betasPred: m.plsC.betas,
        Qpc: m.plsC.q,
        tCV,
        tOrthCV,
        tOrth: m.tOrth,
        pOrth: m.pOrth,
        wOrth: m.wOrth,
        XOrth,
        Yres: m.plsC.yResidual,
        E
      };
    }
    /**
     * get access to all the computed elements
     * Mainly for debug and testing
     * @return {Object} output object
     */


    getLogs() {
      return this.output;
    }

    getScores() {
      let scoresX = this.tCV.map(x => x.to1DArray());
      let scoresY = this.tOrthCV.map(x => x.to1DArray());
      return {
        scoresX,
        scoresY
      };
    }
    /**
     * Load an OPLS model from JSON
     * @param {Object} model
     * @return {OPLS}
     */


    static load(model) {
      if (typeof model.name !== 'string') {
        throw new TypeError('model must have a name property');
      }

      if (model.name !== 'OPLS') {
        throw new RangeError("invalid model: ".concat(model.name));
      }

      return new OPLS(true, [], model);
    }
    /**
     * Export the current model to a JSON object
     * @return {Object} model
     */


    toJSON() {
      return {
        name: 'OPLS',
        center: this.center,
        scale: this.scale,
        means: this.means,
        stdevs: this.stdevs,
        model: this.model,
        tCV: this.tCV,
        tOrthCV: this.tOrthCV
      };
    }
    /**
     * Predict scores for new data
     * @param {Matrix} features - a matrix containing new data
     * @param {Object} [options]
     * @param {Array} [options.trueLabel] - an array with true values to compute confusion matrix
     * @param {Number} [options.nc] - the number of components to be used
     * @return {Object} - predictions
     */


    predict(newData) {
      let options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
      let {
        trueLabels = [],
        nc = 1
      } = options;
      let labels = [];

      if (trueLabels.length > 0) {
        trueLabels = Matrix.from1DArray(trueLabels.length, 1, trueLabels);
        labels = trueLabels.clone();
      }

      let features = newData.clone(); // scaling the test dataset with respect to the train

      if (this.center) {
        features.center('column'); // features.clone().center('column', { center: this.means });
        // if (labels.rows > 0) {

        labels.center('column'); // }
      }

      if (this.scale) {
        features.scale('column'); // features.clone().scale('column', { scale: this.stdevs });
        // if (labels.rows > 0) {

        labels.scale('column'); // }
      }

      let Eh = features.clone(); // removing the orthogonal components from PLS

      let tOrth;
      let wOrth;
      let pOrth;
      let yHat;
      let tPred;

      for (let idx = 0; idx < nc; idx++) {
        wOrth = this.model[idx].wOrth.transpose();
        pOrth = this.model[idx].pOrth;
        tOrth = Eh.clone().mmul(wOrth);
        Eh.sub(tOrth.clone().mmul(pOrth)); // prediction

        tPred = Eh.clone().mmul(this.model[idx].plsC.w.transpose()); // this should be summed over ncomp (pls_prediction.R line 23)

        yHat = tPred.clone().mmul(this.model[idx].plsC.betas);
      } // console.log(yHat);
      // console.log(labels);


      if (labels.rows > 0) {
        if (this.mode === 'regression') {
          let tssy = tss(labels);
          let press = tss(labels.clone().sub(yHat));
          let Q2y = 1 - press / tssy;
          return {
            tPred,
            tOrth,
            yHat,
            Q2y
          };
        } else if (this.mode === 'discriminant_analysis') {
          let confusionMatrix = [];
          confusionMatrix = ConfusionMatrix.fromLabels(trueLabels.to1DArray(), yHat.to1DArray());
          return {
            tPred,
            tOrth,
            yHat,
            confusionMatrix
          };
        }
      } else {
        return {
          tPred,
          tOrth,
          yHat
        };
      }
    }

    _predictAll(features, labels) {
      let options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
      // cannot use the global this.center here
      // since it is used in the NC loop and
      // centering and scaling should only be
      // performed once
      const {
        center = true,
        scale = true
      } = options;

      if (center) {
        features.center('column');
        labels.center('column');
      }

      if (scale) {
        features.scale('column');
        labels.scale('column'); // reevaluate tssy and tssx after scaling
        // must be global because re-used for next nc iteration
        // tssx is only evaluate the first time

        this.tssy = tss(labels);
        this.tssx = tss(features);
      }

      let oplsC = oplsNIPALS(features, labels);
      let plsC = new nipals(oplsC.filteredX, {
        Y: labels
      });
      let tPred = oplsC.filteredX.clone().mmul(plsC.w.transpose());
      let yHat = tPred.clone().mmul(plsC.betas);
      let rss = tss(labels.clone().sub(yHat));
      let R2y = 1 - rss / this.tssy;
      let xEx = plsC.t.clone().mmul(plsC.p.clone());
      let rssx = tss(xEx);
      let R2x = rssx / this.tssx;
      return {
        R2y,
        R2x,
        xRes: oplsC.filteredX,
        tOrth: oplsC.scoresXOrtho,
        pOrth: oplsC.loadingsXOrtho,
        wOrth: oplsC.weightsXOrtho,
        tPred: tPred,
        totalPred: yHat,
        XOrth: oplsC.scoresXOrtho.clone().mmul(oplsC.loadingsXOrtho),
        oplsC,
        plsC
      };
    }
    /**
     *
     * @param {*} X - dataset matrix object
     * @param {*} group - labels matrix object
     * @param {*} index - train and test index (output from getFold())
     */


    _getTrainTest(X, group, index) {
      let testFeatures = new Matrix(index.testIndex.length, X.columns);
      let testLabels = new Matrix(index.testIndex.length, 1);
      index.testIndex.forEach((el, idx) => {
        testFeatures.setRow(idx, X.getRow(el));
        testLabels.setRow(idx, group.getRow(el));
      });
      let trainFeatures = new Matrix(index.trainIndex.length, X.columns);
      let trainLabels = new Matrix(index.trainIndex.length, 1);
      index.trainIndex.forEach((el, idx) => {
        trainFeatures.setRow(idx, X.getRow(el));
        trainLabels.setRow(idx, group.getRow(el));
      });
      return {
        trainFeatures,
        testFeatures,
        trainLabels,
        testLabels
      };
    }

  }

  exports.KOPLS = KOPLS;
  exports.OPLS = OPLS;
  exports.PLS = PLS;
  exports.oplsNIPALS = oplsNIPALS;
  exports.summaryMetadata = summaryMetadata;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=ml-pls.js.map
