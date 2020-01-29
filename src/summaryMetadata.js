import Matrix from 'ml-matrix';

/**
 *
 * @param {*} classVector - an array with class information
 * @return {Object} - an object with class information in different formats
 */

export function summaryMetadata(classVector) {
  let nObs = classVector.length;
  let type = typeof classVector[0];
  let counts = {};
  switch (type) {
    case 'string':
      counts = {};
      classVector.forEach((x) => (counts[x] = (counts[x] || 0) + 1));
      break;
    case 'number':
      classVector = classVector.map((x) => x.toString());
      counts = {};
      classVector.forEach((x) => (counts[x] = (counts[x] || 0) + 1));
      break;
    default:
  }
  let groupIDs = Object.keys(counts);
  let nClass = groupIDs.length;
  let classFactor = classVector.map((x) => groupIDs.indexOf(x));
  let classMatrix = Matrix.from1DArray(nObs, 1, classFactor);
  return { groupIDs, nClass, classVector, classFactor, classMatrix };
}
