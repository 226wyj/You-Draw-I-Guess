var canvas;
var context;

// 初始化
window.onload = function() {
  // 获取画布已经绘图上下文
  canvas = document.getElementById("drawingCanvas");
  context = canvas.getContext("2d");

  // 画布添加鼠标事件
  canvas.onmousedown = startDrawing;
  canvas.onmouseup = stopDrawing;
  canvas.onmouseout = stopDrawing;
  canvas.onmousemove = draw;
};

// 记录当前是否在画图
var isDrawing = false;

// 开始画图
function startDrawing(e) {
  isDrawing = true;
  // 创建一个新的绘图路径
  context.beginPath();
  // 把画笔移动到鼠标位置
  context.moveTo(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop);
}

// 停止画图
function stopDrawing() {
  isDrawing = false;
}

//画图中
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

// 保存之前选择的颜色的画笔 <img> 元素标签
var previousColorElement;

// 改变画笔颜色
function changeColor(color, imgElement) {    
  context.strokeStyle = color;
  // 将当前画笔的 <img> 元素标签设置为选中样式
  imgElement.className = "Selected";
  // 将之前的画笔的 <img> 元素标签恢复默认样式
  if (previousColorElement != null) previousColorElement.className = "";
  previousColorElement = imgElement;
}

// 保存之前选择的粗细的画笔 <img> 元素标签
var previousThicknessElement;

// 改变画笔粗细
function changeThickness(thickness, imgElement) {    
  context.lineWidth = thickness;
  // 将当前画笔的 <img> 元素标签设置为选中样式
  imgElement.className = "Selected";
  // 将之前的画笔的 <img> 元素标签恢复默认样式
  if (previousThicknessElement != null) previousThicknessElement.className = "";
  previousThicknessElement = imgElement;
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
