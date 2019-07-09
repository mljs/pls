import Matrix from 'ml-matrix';
import Stat from 'ml-stat';


/**
 * @private
 * Function that given vector, returns its norm
 * @param {Vector} X
 * @return {number} Norm of the vector
 */
export function norm(X) {
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
export function pow2array(i, j) {
  this.set(i, j, this.get(i, j) ** 2);
}

/**
 * @private
 * Function that normalize the dataset and return the means and
 * standard deviation of each feature.
 * @param {Matrix} dataset
 * @return {object} dataset normalized, means and standard deviations
 */
export function featureNormalize(dataset) {
  var means = dataset.mean('column');
  var std = dataset.standardDeviation('column', { mean: means, unbiased: true });
  var result = Matrix.checkMatrix(dataset).subRowVector(means);
  return { result: result.divRowVector(std), means: means, std: std };
}

/**
 * @private
 * Function that initialize an array of matrices.
 * @param {Array} array
 * @param {boolean} isMatrix
 * @return {Array} array with the matrices initialized.
 */
export function initializeMatrices(array, isMatrix) {
  if (isMatrix) {
    for (var i = 0; i < array.length; ++i) {
      for (var j = 0; j < array[i].length; ++j) {
        var elem = array[i][j];
        array[i][j] = elem !== null ? new Matrix(array[i][j]) : undefined;
      }
    }
  } else {
    for (i = 0; i < array.length; ++i) {
      array[i] = new Matrix(array[i]);
    }
  }

  return array;
}

/**
 * @private
 * Get total sum of square
 * @param {Array} x an array
 */
export function tss(x) {
  return x.mul(x).sum();
}

/**
 * @private
 * Compute Q2 statistics
 * @param {Array} realY an array with real/true values
 * @param {Array} predictedY an array with predicted values
 * @return {Number} Q2 statistics
 */
export function Q2(realY, predictedY) {
  realY = Matrix.checkMatrix(realY);
  predictedY = Matrix.checkMatrix(predictedY);
  let meansY = Stat.array.mean(realY);

  let press = predictedY.map((row, rowIndex) => {
    return row.map((element, colIndex) => {
      return Math.pow(realY[rowIndex][colIndex] - element, 2);
    });
  });


  let tss = Y.map((row) => {
    return row.map((element, colIndex) => {
      return Math.pow(element - meansY[colIndex], 2);
    });
  });

  press = Matrix.checkMatrix(press).sum();
  tss = Matrix.checkMatrix(tss).sum();

  return 1 - press / tss;
}


/**
 * @private
 * scales a dataset
 * @param {*} dataset
 * @param {*} options
 */
export function scale(dataset, options) {
  let center = !!options.center;
  let scale = !!options.scale;

  dataset = new Matrix(dataset);

  if (center) {
    const means = dataset.mean('column');
    const stdevs = scale ? dataset.standardDeviation('column') : null;
    // means = means;
    dataset.subRowVector(means);
    if (scale) {
      for (var i = 0; i < stdevs.length; i++) {
        if (stdevs[i] === 0) {
          throw new RangeError(`Cannot scale the dataset (standard deviation is zero at index ${i}`);
        }
      }
      // stdevs = stdevs;
      dataset.divRowVector(stdevs);
    }
  }

  return dataset;
}

/**
 * @private
 * create a dataset from data
 * @param {Object} object with parameters
 * @return {Object} dataset object
 */

export const Dataset = ({ dataMatrix, options } = {}) => {
  let nObs = dataMatrix.rows;
  let nVar = dataMatrix.columns;

  let aa = {};
  const {
    descriptio = 123
    // observations = Array(nObs).fill(null).map((x, i) => 'OBS' + (i + 1)),
    // variables = Array(nVar).fill(null).map((x, i) => 'VAR' + (i + 1)),
    // description = 'NA'
    // metadata = [],
    // outliers = []
  } = aa;

  let defaults = {
    observations: Array(nObs).fill(null).map((x, i) => `OBS${i + 1}`),
    variables: Array(nVar).fill(null).map((x, i) => `VAR${i + 1}`),
    description: 'NA',
    metadata: [],
    outliers: []
  };

  options = Object.assign({}, defaults, options);

  let observations = options.observations;
  let variables = options.variables;
  let description = options.description;
  let dataClass = options.dataClass;
  let metadata = options.metadata;
  let outliers = options.outliers;

  if (options.observations.length !== nObs ||
        options.variables.length !== nVar ||
        options.dataClass[0].value.length !== nObs) {
    throw new RangeError('observations and dataMatrix have different number of rows');
  }

  // private util functions

  function getRowIndexByID() {
    let sampleList = observations.map((x) => x);
    let outlierList = outliers.map((x) => x.id);
    let ind = outlierList.map((e) => sampleList.indexOf(e));
    return ind;
  }

  function getClassVector(dataClass) {
    let title = dataClass.title;
    let classVector = dataClass.value;
    let type = typeof (classVector[0]);
    let counts = {};
    switch (type) {
      case 'string':
        counts = {};
        classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
        break;
      case 'number':
        classVector = classVector.map((x) => x.toString());
        counts = {};
        classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
        break;
      default:
    }
    let groupIDs = Object.keys(counts);
    let nClass = groupIDs.length;
    let classFactor = classVector.map((x) => groupIDs.indexOf(x));
    let classMatrix = Matrix.from1DArray(nObs, 1, classFactor);
    return ({ title,
      groupIDs,
      nClass,
      classVector,
      classFactor,
      classMatrix
    });
  }

  return ({ description,

    // API exposed functions

    getClass() {
      let a = dataClass.map((x) => getClassVector(x));
      return a;
    },

    getOutliers() {
      return outliers;
    },

    addOutliers(outliersList) {
      let outliersIDs = outliers.map((x) => x.id);
      let newList = outliersList.filter((f) => !outliersIDs.includes(f.id));
      console.log(newList);
      if (newList.length > 0) newList.forEach((e) => outliers.push(e));
      console.log(outliers);
      this.summary();
      // return this;
    },

    rmOutliers(outliersList) {
      outliers = outliers.filter((f) => !outliersList.includes(f.id));
      console.log(outliers);
      this.summary();
      // return this;
    },

    clean() {
      if (outliers.length > 0) {
        let ind = getRowIndexByID();
        console.log(ind);
        let cleanObservations = observations.filter((e, i) => !ind.includes(i));
        let cleanDataMatrix = new Matrix(nObs - ind.length, nVar);

        let counter = 0;
        dataMatrix.forEach((e, i) => {
          if (!ind.includes(i)) {
            cleanDataMatrix.setRow(counter, e);
            counter += 1;
          }
        });

        let cleanDataClass = dataClass.map((x) => {
          return { title: x.title,
            value: x.value.filter((e, i) => !ind.includes(i)) };
        });
        // let cleanMetadata = metadata.filter((e, i) => !ind.includes(i));

        return Dataset({
          dataMatrix: cleanDataMatrix,
          options: {
            observations: cleanObservations,
            variables: variables,
            dataClass: cleanDataClass,
            outliers: [],
            // metadata: cleanMetadata, // lack of test for dimensions
            description: `clean ${description}` } });
      } else {
        return this;
      }
    },

    sample(list) {
      // console.log(list.length)
      if (list.length > 0) {
        let ind = list;
        // console.log(ind);
        // filter Observations vector

        let trainObservations = observations.filter((e, i) => !ind.includes(i));
        let testObservations = observations.filter((e, i) => ind.includes(i));

        // filter data matrix
        let trainDataMatrix = new Matrix(nObs - ind.length, nVar);
        let testDataMatrix = new Matrix(ind.length, nVar);

        let counter = 0;
        dataMatrix.forEach((e, i) => {
          if (!ind.includes(i)) {
            trainDataMatrix.setRow(counter, e);
            counter += 1;
          }
        });

        counter = 0;
        dataMatrix.forEach((e, i) => {
          if (ind.includes(i)) {
            testDataMatrix.setRow(counter, e);
            counter += 1;
          }
        });

        // filter class vector
        let trainDataClass = dataClass.map((x) => {
          return { title: x.title,
            value: x.value.filter((e, i) => !ind.includes(i)) };
        });

        let testDataClass = dataClass.map((x) => {
          return { title: x.title,
            value: x.value.filter((e, i) => ind.includes(i)) };
        });

        // let cleanMetadata = metadata.filter((e, i) => !ind.includes(i));

        let train = Dataset({
          dataMatrix: trainDataMatrix,
          options: {
            observations: trainObservations,
            variables: variables,
            dataClass: trainDataClass,
            outliers: [],
            // metadata: cleanMetadata, // lack of test for dimensions
            description: `train ${description}` } });

        let test = Dataset({
          dataMatrix: testDataMatrix,
          options: {
            observations: testObservations,
            variables: variables,
            dataClass: testDataClass,
            outliers: [],
            // metadata: cleanMetadata, // lack of test for dimensions
            description: `test ${description}` } });

        return { train, test };
      } else {
        return this;
      }
    },

    // return everything but cannot be changed
    summary(verbose = 0) {
      if (verbose === 1) {
        console.log(`Description: ${description
        }\nNumber of variables: ${nVar
        }\nNumber of observations: ${nObs
        }\nNumber of outliers:${outliers.length
        }\nHas class: ${dataClass.length
        }\nHas metadata: ${metadata.length > 0}`);
      }
      return ({ dataMatrix,
        dataClass,
        nObs,
        nVar,
        observations,
        variables,
        metadata,
        description
      });
    }

  });
};

/* *
 * @private
 * Shuffles array for permutation (from ml knn.js)
 * @param {Array} array
 */
/* export function shuffleArray(array) {
  for (var i = array.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
  return array;
} */

/**
 * @private
 * create a class vector
 * @param {String} title a title for the class
 * @param {*} value
 */
/* export const DataClass = (title, value) => {
  let dataClasses = [];
  dataClasses.push({ title, value });
  return ({
    getClass() {
      return dataClasses;
    },
    addClass(title, value) {
      dataClasses.push({ title, value });
      return this;
    }
  });
}; */


export function sampleAClass(classVector, fraction) {
  // sort the vector
  let classVectorSorted = JSON.parse(JSON.stringify(classVector));
  let result = Array.from(Array(classVectorSorted.length).keys())
    .sort((a, b) => (classVectorSorted[a] < classVectorSorted[b] ? -1 :
      (classVectorSorted[b] < classVectorSorted[a]) | 0));
  classVectorSorted.sort((a, b) => (a < b ? -1 : (b < a) | 0));

  // counts the class elements
  let counts = {};
  classVectorSorted.forEach((x) => counts[x] = (counts[x] || 0) + 1);

  // pick a few per class
  let indexOfSelected = [];

  Object.keys(counts).forEach((e, i) => {
    let shift = [];
    Object.values(counts).reduce((a, c, i) => shift[i] = a + c, 0);

    let arr = [...Array(counts[e]).keys()];

    let r = [];
    for (let i = 0; i < Math.floor(counts[e] * fraction); i++) {
      let n = arr[Math.floor(Math.random() * arr.length)];
      r.push(n);
      let ind = arr.indexOf(n);
      arr.splice(ind, 1);
    }

    (i == 0) ? indexOfSelected = indexOfSelected.concat(r) : indexOfSelected = indexOfSelected.concat(r.map((x) => x + shift[i - 1]));
  });

  // sort back the index
  let trainIndex = [];
  indexOfSelected.forEach((e) => trainIndex.push(result[e]));

  let testIndex = [];
  let mask = [];
  classVector.forEach((el, idx) => {
    if (trainIndex.includes(idx)) {
      mask.push(true);
    } else {
      mask.push(false);
      testIndex.push(idx);
    }
  });
  return { trainIndex, testIndex, mask };
}

export function summaryMetadata(classVector) {
  let nObs = classVector.length;
  let type = typeof (classVector[0]);
  let counts = {};
  switch (type) {
    case 'string':
      counts = {};
      classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
      break;
    case 'number':
      classVector = classVector.map((x) => x.toString());
      counts = {};
      classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
      break;
    default:
  }
  let groupIDs = Object.keys(counts);
  let nClass = groupIDs.length;
  let classFactor = classVector.map((x) => groupIDs.indexOf(x));
  let classMatrix = Matrix.from1DArray(nObs, 1, classFactor);
  return ({ groupIDs,
    nClass,
    classVector,
    classFactor,
    classMatrix
  });
}

/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Array} labels - an aray with class/groups/labels
  * */
export class METADATA {
  constructor(metadata, options = {}) {
    if (metadata === true) {
      const model = options;
      this.center = model.center;
      this.scale = model.scale;
      this.means = model.means;
      this.stdevs = model.stdevs;
      this.U = Matrix.checkMatrix(model.U);
      this.S = model.S;
      return;
    }

    const {
      isCovarianceMatrix = false,
      center = true,
      scale = false
    } = options;

    this.metadata = [];
  }
  /**
   * listMetadata
   */
  listMetadata() {
    return this.metadata.map((x) => x.title);
  }
  /**
   * add metadata
   * @param {String} title - a title
   * @param {Array} value - an array with metadata
   */
  addMetadata(title, value) {
    this.metadata.push({ title, value });
    return this;
  }
  /**
   *
   * @param {String} title - a title
   * @return {Object} return { title, groupIDs, nClass, classVector, classFactor, classMatrix }
   */
  getMetadata(title) {
    let classVector = this.metadata.filter((x) => x.title === title)[0].value;
    let nObs = classVector.length;
    let type = typeof (classVector[0]);
    let counts = {};
    switch (type) {
      case 'string':
        counts = {};
        classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
        break;
      case 'number':
        classVector = classVector.map((x) => x.toString());
        counts = {};
        classVector.forEach((x) => counts[x] = (counts[x] || 0) + 1);
        break;
      default:
    }
    let groupIDs = Object.keys(counts);
    let nClass = groupIDs.length;
    let classFactor = classVector.map((x) => groupIDs.indexOf(x));
    let classMatrix = Matrix.from1DArray(nObs, 1, classFactor);
    return ({ title,
      groupIDs,
      nClass,
      classVector,
      classFactor,
      classMatrix
    });
  }
}
