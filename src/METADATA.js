import Matrix from 'ml-matrix';
import CV from 'ml-cross-validation';

import { plsNIPALS } from './plsNIPALS.js';
import { oplsNIPALS } from './oplsNIPALS.js';
import { tss, sampleAClass, summaryMetadata, getFolds } from './utils.js';


/**
 * Creates new PCA (Principal Component Analysis) from the dataset
 * @param {Array} labels - an aray with class/groups/labels
  * */
export class METADATA {
  constructor(metadata, options = {}) {
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
