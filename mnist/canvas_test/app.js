const btnReset = document.getElementById("btnReset");
const btnSave = document.getElementById("btnSave");

const canvas = document.getElementById("MNIST");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "#000000";
ctx.fillRect(0, 0, 700, 700);

const icanvas = document.getElementById("input");
const ictx = icanvas.getContext("2d");
ictx.fillStyle = "#000000";
ictx.fillRect(0, 0, 700, 700);

ctx.strokeStyle = "#ffffff";
ctx.lineWidth = 20;

let painting = false;

function stopPainting(){
    painting = false;
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

function cvtCanvasToImage()
{
    let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let src = cv.matFromImageData(imgData);
    cv.resize(src, src, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);
    cv.imshow(icanvas, src);
}

function handleResetClick()
{
    ctx.fillRect(0, 0, 700, 700);
    cvtCanvasToImage();
	document.getElementById("result").innerHTML = "예측값 : ";
}

async function handleSaveClick()
{
    cvtCanvasToImage();

    const tensor = tf.browser
    .fromPixels(canvas, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .expandDims();
    console.log("input shape : " + tensor.shape);

    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json');
    console.log(model.summary());

	const output = model.predict(tensor);
	const predictions = output.argMax(1).dataSync()[0];
    const confidence = output.max(1).dataSync()[0];

    const out = predictions + "(" + confidence + ")";
    console.log("======================== 예측값 : " + out);

	document.getElementById("result").innerHTML = "예측값 : " + out;
}

if(canvas)
{
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", stopPainting);
    canvas.addEventListener("mouseleave", stopPainting);
}

if(btnReset) btnReset.addEventListener("click", handleResetClick);
if(btnSave) btnSave.addEventListener("click", handleSaveClick);