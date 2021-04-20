var canvas;
var context;

// Initialize
window.onload = function() {
	// Get canvas and the picture's context.
	canvas = document.getElementById("drawingCanvas");
	context = canvas.getContext("2d");

	// Add mouse action listener to canvas.
	canvas.onmousedown = startDrawing;
	canvas.onmouseup = stopDrawing;
	canvas.onmouseout = stopDrawing;
	canvas.onmousemove = draw;
};

// Judge whether is drawing.
var isDrawing = false;

// Start drawing.
function startDrawing(e) {
	isDrawing = true;
	// 创建一个新的绘图路径
	context.beginPath();
	// 把画笔移动到鼠标位置
	context.moveTo(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop);
}

// Stop drawing.
function stopDrawing() {
  	isDrawing = false;
}

// 画图中
function draw(e) {
  if (isDrawing == true) {
	// 找到鼠标最新位置
	var x = e.pageX - canvas.offsetLeft;
	var y = e.pageY - canvas.offsetTop;
	// 画一条直线到鼠标最新位置
	context.lineTo(x, y);
	context.stroke();  
  }
}

// 保存之前选择的颜色的画笔 <button> 元素标签
var previousColorElement;

// 改变画笔颜色
function changeColor(color, btnElement) {    
	context.strokeStyle = color;
	// 将当前画笔的 <button> 元素标签设置为选中样式
	btnElement.className = "btn btn-primary active";
	// 将之前的画笔的 <button> 元素标签恢复默认样式
	if (previousColorElement != null) 
		previousColorElement.className = "btn btn-default";
	previousColorElement = btnElement;
}

// 保存之前选择的粗细的画笔 <button> 元素标签
var previousThicknessElement;

// 改变画笔粗细
function changeThickness(thickness, btnElement) {    
	context.lineWidth = thickness;
	// 将当前画笔的 <button> 元素标签设置为选中样式
	btnElement.className = "btn btn-primary active";
	// 将之前的画笔的 <button> 元素标签恢复默认样式
	if (previousThicknessElement != null)
		previousThicknessElement.className = "btn btn-default";
	previousThicknessElement = btnElement;
}

// 清除画布
function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

// 保存画布
function saveCanvas() {
  // 找到预览的 <img> 元素标签
  var imageCopy = document.getElementById("savedImageCopy");
  // 将画布内容在img元素中显示
  imageCopy.src = canvas.toDataURL(); 
  // 显示右键保存的提示
  var imageContainer = document.getElementById("savedCopyContainer");
  imageContainer.style.display = "block";
}
