// ==UserScript==
// @name        NTHU OAuth Decaptcha Local
// @namespace   https://github.com/koihia/NTHU-OAuth-Decaptcha
// @match       https://oauth.ccxp.nthu.edu.tw/*
// @grant       GM_xmlhttpRequest
// @require     https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.15.0/dist/tf.min.js
// @version     1.0
// @author      koihia
// @description Automatically fill in the captcha code in the NTHU OAuth login page locally using TensorFlow.js.
// ==/UserScript==
"use strict";

(async () => {
  const modelUrl =
    "https://raw.githubusercontent.com/koihia/NTHU-OAuth-Decaptcha/master/userscript-local/model/model.json";

  const insertCode = (code) => {
    const captchaInput = document.getElementById("captcha_code");
    captchaInput.value = code;
  };

  const predict = (model, captchaImage) => {
    const prediction = tf.tidy(() => {
      const imgTensor = tf.browser
        .fromPixels(captchaImage)
        .div(tf.scalar(255))
        .expandDims();

      const prediction = model.predict(imgTensor);

      return prediction;
    });

    const axis = 1;
    let answer = "";
    for (let i = 0; i < 4; i++) {
      answer += prediction[i].argMax(axis).dataSync();
    }

    return answer;
  };

  const run = (model, captchaImage) => {
    const code = predict(model, captchaImage);
    insertCode(code);
  };

  const model = await tf.loadLayersModel(modelUrl);
  const captchaImage = document.getElementById("captcha_image");
  run(model, captchaImage);
  captchaImage.addEventListener("load", () => run(model, captchaImage));
})();
