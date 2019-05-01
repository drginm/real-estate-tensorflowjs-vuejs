<template>
  <div>
    <div class="train-controls">
      <h2 class="section-header">House/Apartment parameters</h2>

      <div>
        <div class="col-sm-12">
          <div class="col-sm-6 field-label">Size (mts)</div>
          <input class="col-sm-6 field field-x"
                v-model="size"
                type="number">

          <div class="col-sm-6 field-label">Rooms</div>
          <input class="col-sm-6 field field-x"
                v-model="rooms"
                type="number">

          <div class="col-sm-6 field-label">Baths</div>
          <input class="col-sm-6 field field-x"
                v-model="baths"
                type="number">

          <div class="col-sm-6 field-label">Parking</div>
          <input class="col-sm-6 field field-x"
                v-model="parking"
                type="number">

          <div class="col-sm-6 field-label">Neighborhood</div>
          <select class="col-sm-6 neighborhood field" v-model="neighborhood">
            <option disabled value="">Please select one</option>
            <option v-for="(item, index) in neighborhoods" v-bind:key="index" v-html="item" :value="index"></option>
          </select>
        </div>
      </div>
    </div>

    <div class="predict-controls">
      <h2 class="section-header">Predicting</h2>
      <div class="col-sm-6 field-label">Predicted Value</div>
      <div class="col-sm-6 element field" v-html="predictedValue"></div>
      <button class="col-sm-12 element button--green" v-on:click="predict" :disabled="!modelReady">Predict</button>
    </div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs';
import * as onehot from 'one-hot-enum';

import neighborhoods from '~/static/shared/neighborhoods.json'

import meanX from '~/static/shared/scaler-mean-x.json'
import varX from '~/static/shared/scaler-var-x.json'

import meanY from '~/static/shared/scaler-mean-y.json'
import varY from '~/static/shared/scaler-var-y.json'

export default {
  data() {
    return {
      modelReady: false,
      size:180,
      rooms: 5,
      baths: 2,
      parking: 0,
      neighborhood: 0,
      predictedValue:'Model not loaded!',
      selected: '',
      neighborhoods: neighborhoods
    }
  },
  mounted() {
    this.initializeScaler();

    this.initializeOneHotEncoder();

    this.loadModel();
  },
  methods: {
    //Initialization
    initializeScaler() {
      this.meanX = tf.tensor1d(meanX);
      this.deviationX = tf.tensor1d(varX).sqrt();

      this.meanY = tf.tensor1d(meanY);
      this.deviationY = tf.tensor1d(varY).sqrt();
    },
    initializeOneHotEncoder() {
      let reducedlist = this.neighborhoods.slice(1);
      let enumaration = onehot.enumaration(reducedlist);
      let encoded = onehot.encode(reducedlist);
      let zeros = Array.apply(null, Array(encoded[0].length)).map(Number.prototype.valueOf, 0);

      this.dictionary = {};

      for (let i in enumaration) {
        this.dictionary[enumaration[i]] = encoded[i];
      }

      this.dictionary[this.neighborhoods[0]] = zeros;
    },
    async loadModel() {
      this.model = await tf.loadLayersModel('shared/model/model.json');

      this.modelReady = true;
      this.predictedValue = 'Ready for making predictions';
    },
    //Prediction
    scale(value, mean, deviation) {
      //(value - mean) / Math.sqrt(variance);
      return value.sub(mean)
                  .div(deviation);
    },
    unscale(value, mean, deviation) {
      //(value * Math.sqrt(variance)) + mean;
      return value.mul(deviation)
                  .add(mean);
    },
    preProcessInputs(inputs, neighborhood) {
      let modelInput = tf.tensor1d(inputs);
      let neighborhoodTensor = tf.tensor1d(this.dictionary[this.neighborhoods[neighborhood]]);

      modelInput = this.scale(modelInput, this.meanX, this.deviationX);

      modelInput = modelInput.concat(neighborhoodTensor)
                             .expandDims();

      return modelInput;
    },
    postProcessResults(outputs) {
      return this.unscale(outputs, this.meanY, this.deviationY);
    },
    predict() {
      //Transform Inputs
      let modelInput = this.preProcessInputs([
                                               parseFloat(this.size),
                                               parseFloat(this.rooms),
                                               parseFloat(this.baths),
                                               parseFloat(this.parking)
                                             ], this.neighborhood);

      //Get prediction
      const prediction = this.model.predict(modelInput);
      window.myVal = this.postProcessResults(prediction);

      //Transform Outputs
      this.predictedValue = Math.ceil(this.postProcessResults(prediction).dataSync()[0]);

      console.log(this.neighborhoods[this.neighborhood], prediction.dataSync()[0], this.predictedValue);
    }
  }
}
</script>

<style>
.neighborhood {
  text-transform: capitalize;
}
.field, .field-label {
  height: 30px;
  float: left;
  padding: 0px 15px;
}

.field {
  border-radius: 0px 5px 5px 0px;
  border: 1px solid #eee;
  margin-bottom: 15px;
  height: 40px;
}

.col-sm-1:after {
    content: "";
    display: table;
    clear: both;
}

.section-header, .field-label {
  text-align: left;
  font-family: "Quicksand", "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; /* 1 */
  font-weight: 100;
}

.field-label {
  font-weight: 700;
}

.predict-controls {
  padding-top: 30px;
  padding-bottom: 30px;
}

.predict-controls .element {
  display: block;
}

button {
  margin-top: 10px;
  font-family: "Quicksand", "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; /* 1 */
  font-weight: 700;
}

</style>
