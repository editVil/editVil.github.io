// https://medium.com/ailab-telu/learn-and-play-with-tensorflow-js-part-3-dd31fcab4c4b

// const { string } = require("@tensorflow/tfjs");
// ;(async() => { await tf.setBackend('cpu'); })();
// console.log(tf.getBackend());

const canvas = document.getElementById("handWrite");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "#000000";
ctx.fillRect(0, 0, 700, 700);
ctx.strokeStyle = "#ffffff";
ctx.lineWidth = 20;
let painting = false;

const imgTest1 = document.getElementById("imgTest1");
const imgTest2 = document.getElementById("imgTest2");
const imgTest3 = document.getElementById("imgTest3");
const imgTest4 = document.getElementById("imgTest4");
const imgTest5 = document.getElementById("imgTest5");

const btnLoadData = document.getElementById("btnLoadData");
const btnBrowseJson = document.getElementById("btnBrowseJson");
const btnBrowseWeight = document.getElementById("btnBrowseWeight");
const btnLoadModel = document.getElementById("btnLoadModel");
const btnTrainModel = document.getElementById("btnTrainModel");
const btnSaveModel = document.getElementById("btnSaveModel");
const btnEvalModel = document.getElementById("btnEvalModel");
const btnReset = document.getElementById("btnReset");
const btnPrediction = document.getElementById("btnPrediction");
const btnGoogleModel = document.getElementById("btnGoogleModel");
const btnCNNModel = document.getElementById("btnCNNModel");

const strJsonFile = document.getElementById("strJsonFile");
const strWeightFile = document.getElementById("strWeightFile");

const txtSummary = document.getElementById("txtSummary");
if(txtSummary) txtSummary.disabled = true;

const txtEpoch = document.getElementById("txtEpoch");
const txtBatchSize = document.getElementById("txtBatchSize");
const strNumIteration = document.getElementById("strNumIteration");

let model;

var data = new MnistData();

function stopPainting(){
    painting = false;
}

function cvtCanvasToImage()
{
    let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let src = cv.matFromImageData(imgData);
    cv.resize(src, src, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);
    return src;
}

function onMouseMove(event){
    const x = event.offsetX;
    const y = event.offsetY;

    if(!painting) {
        ctx.beginPath();
        ctx.moveTo(x, y);
    }
    else{
        ctx.lineTo(x, y);
        ctx.stroke();
    }
}

function onMouseDown(event){
    painting = true;
    if(event.button === 0) ctx.strokeStyle = "#ffffff";
    else ctx.strokeStyle = "#000000";
}

function getFileNameFromPath(path) {
    return path.substring(path.lastIndexOf('/')+1);
}

function getStringSummary(model){
    if(txtSummary) {
        txtSummary.value = "[layer name     {layer type     }]     input                         output                        params\n";
        txtSummary.value += "=============================================================================================================================\n";

        let allParams = 0;
        var layers = model.layers;
        for(var i = 0; i < layers.length ; i++){
            let outputShape = "";
            let inputShape = "";
            let layer = layers[i];

            try {
                inputShape = (layer.inboundNodes.map(
                x => JSON.stringify(x.inputShapes)
                )).join(',');
            } catch (err) {
                inputShape = 'multiple';
            }

            try {
                outputShape = JSON.stringify(layer.outputShape);
            } catch (err) {
                outputShape = 'multiple';
            }

            const params = layer.countParams();
            const name = (layer.name).padEnd(15, ' ').substr(0, 15);
            const className = layer.getClassName().padEnd(15, ' ').substr(0, 15);
            inputShape = inputShape.padEnd(30, ' ').substr(0, 30);
            outputShape = outputShape.padEnd(30, ' ').substr(0, 30);
            txtSummary.value += "[" + name + "{" + className + "}]     " + inputShape + outputShape + params.toString() + "\n";
            txtSummary.value += "------------------------------------------------------------------------------------------------------------------------------\n";
            allParams += params;
        }
        txtSummary.value += allParams.toString().padStart(105, ' ');
    }

    txtSummary.style.height = txtSummary.scrollHeight - 4 + 'px';
}

async function update(model) {
    getStringSummary(model);
}

function btnUpdateState_tain(bDisable){
    btnLoadData.disabled = bDisable;
    if(document.getElementById("inputJson").files && document.getElementById("inputWeight").files) btnLoadModel.disabled = bDisable;
    btnTrainModel.disabled = bDisable;
    btnSaveModel.disabled = bDisable;
    btnEvalModel.disabled = bDisable;
    btnReset.disabled = bDisable;
    btnPrediction.disabled = bDisable;
}

async function btnClicked_LoadDataClick(){
    console.log("Downloading MNIST data. Please wait...");
    await data.load(40000, 10000)
    
    console.log("Loading MNIST Test data....");
    const [x_test, y_test] = data.getTestData(5)
    const labels = Array.from(y_test.argMax(1).dataSync())
    console.log("Success!!");

    showExample("imgTest", x_test, labels);

    btnTrainModel.disabled = false;
    btnSaveModel.disabled = false;
    btnEvalModel.disabled = false;
}

function btnClicked_btnBrowseWeight(){
    
}

async function btnClicked_btnLoadModel(){
    const jsonUpload = document.getElementById("inputJson").files[0];
    const weightsUpload = document.getElementById("inputWeight").files[0];
    model = await tf.loadLayersModel(tf.io.browserFiles([jsonUpload, weightsUpload]))
    update(model);
}

async function btnClicked_btnTrainModel(){
    btnUpdateState_tain(true);
    try{
        console.log("Training, please wait...");
        var epoch = parseInt(txtEpoch.value);
        var batch = parseInt(txtBatchSize.value);
        
        const [x_train, y_train] = data.getTrainData();
        const [x_test, y_test] = data.getTestData(100);
        console.log("Training, parameter initialized...");
        
        let nIter = 0;
        const numIter = Math.ceil(x_train.shape[0] / batch) * epoch;
        strNumIteration.innerText = "Num Training Iteration: " + numIter.toString() + " (0%)";
            
        const trainLogs = []
        const loss = document.getElementById("loss-graph");
        const acc = document.getElementById("acc-graph");
    
        console.log("Training, preparing...");
        const myOptim = 'rmsprop';
        model.compile( { loss: 'categoricalCrossentropy', optimizer: myOptim, metrics:['accuracy'] } );
    
        console.log("Training...");
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const history = await model.fit(x_train, y_train, {
            batchSize: batch,
            validationData: [x_test, y_test],
            epochs: epoch,
            shuffle: true,
            callbacks: {
                onBatchEnd: async (batch, logs) => {
                    nIter++
                    trainLogs.push(logs)
                    tfvis.show.history(loss, trainLogs, ['loss'], { width: 600, height: 300 })
                    tfvis.show.history(acc, trainLogs, ['acc'], { width: 600, height: 300 })
                    strNumIteration.innerText = "Num Training Iteration: " + numIter.toString() + " (" + Math.round((nIter / numIter)*100) + "%)";
                    // $('#train-acc').text('Training Accuracy : '+ round(logs.acc) +'%')
                },
            }
        });
    
        // $('#train-iter').toggleClass('badge-warning badge-success')
        console.log("Training Done");
    }catch{

    }
    
    btnUpdateState_tain(false);
}

async function btnClicked_btnSaveModel()
{
    const saveResults = await model.save('downloads://')
}

async function btnClicked_btnEvalModel()
{
    console.log("Evaluate, please wait...");
    let [x_test, y_test] = data.getTestData();
    
    let y_pred = model.predict(x_test).argMax(1);
    let y_label = y_test.argMax(1);
    console.log("Evaluate, data load...");

    let eval_test = await tfvis.metrics.accuracy(y_label, y_pred);
    document.getElementById("test-acc").innerText = 'Testset Accuracy : ' + eval_test;
    
    const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
    const conf = document.getElementById("confusion-matrix");
    const acc = document.getElementById("class-accuracy");
    
    const classaAcc = await tfvis.metrics.perClassAccuracy(y_label, y_pred);
    const confMt = await tfvis.metrics.confusionMatrix(y_label, y_pred);

    tfvis.show.perClassAccuracy(acc, classaAcc, classNames);
    tfvis.render.confusionMatrix(conf, { values: confMt , tickLabels: classNames } , {width: 450});
    console.log("Evaluate Done");

    y_label.dispose();
}

function btnClicked_btnReset(){
    ctx.fillRect(0, 0, 700, 700);
}

function btnClicked_btnPrediction(){
    const x_data = tf.browser
    .fromPixels(canvas, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .expandDims();
    var y_pred = model.predict(x_data);
    var prediction = Array.from(y_pred.argMax(1).dataSync());
    document.getElementById("prediction").innerText = "Predicted : "+ prediction;
    
    const barchartData = Array.from(y_pred.dataSync()).map((d, i) => {
        return { index: i, value: d }
    });
    tfvis.render.barchart(document.getElementById("predict-graph"), barchartData,  { width: 600, height: 300 });
}

function onChangeInput(){
    if(document.getElementById("inputJson").files && document.getElementById("inputWeight").files)
        btnLoadModel.disabled = false;
}

async function btnClicked_btnGoogleModel() {
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json');
    update(model);
}

function btnClicked_btnCNNModel() {
    var script = getModel("CNN");
    model = tf.sequential();
    eval(script);
    update(model);
}

if(btnLoadData) btnLoadData.addEventListener("click", btnClicked_LoadDataClick);
if(btnBrowseJson) btnBrowseJson.addEventListener("click", btnClicked_btnBrowseJson);
if(btnBrowseWeight) btnBrowseWeight.addEventListener("click", btnClicked_btnBrowseWeight);
if(btnLoadModel) {
    btnLoadModel.addEventListener("click", btnClicked_btnLoadModel);
    btnLoadModel.disabled = true;
}
if(btnTrainModel) {
    btnTrainModel.addEventListener("click", btnClicked_btnTrainModel);
    btnTrainModel.disabled = true;
}
if(btnSaveModel) {
    btnSaveModel.addEventListener("click", btnClicked_btnSaveModel);
    btnSaveModel.disabled = true;
}
if(btnEvalModel) {
    btnEvalModel.addEventListener("click", btnClicked_btnEvalModel);
    btnEvalModel.disabled = true;
}
if(btnReset) btnReset.addEventListener("click", btnClicked_btnReset);
if(btnPrediction) btnPrediction.addEventListener("click", btnClicked_btnPrediction);
if(btnGoogleModel) btnGoogleModel.addEventListener("click", btnClicked_btnGoogleModel);
if(btnCNNModel) btnCNNModel.addEventListener("click", btnClicked_btnCNNModel);

if(canvas){
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", stopPainting);
    canvas.addEventListener("mouseleave", stopPainting);
}

;(async() => {
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json');
    update(model);
})();
