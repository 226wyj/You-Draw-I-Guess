import sys
from streamlit.proto.Image_pb2 import Image
sys.path.append('..')

from streamlit.proto.RootContainer_pb2 import SIDEBAR
import streamlit as st
from predict import Predictor
from models.lenet import LeNet
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from models.vgg16 import Vgg16_Net
import torch as t
import os
from PIL import Image

with open('style.css', 'r') as f:
    st.markdown('<style>{}<style>'.format(f.read()), unsafe_allow_html=True)

# all  classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# show necessary messages
st.title("Image Classification System")
st.subheader("This is the final project of our team, and the members are shown below:")

st.markdown(
    """
    <div align="center">
    <table>
        <tr>
            <th>Name</th>
            <th>NetID</th>
        </tr>
        <tr>
            <td>Yuejiang Wu</td>
            <td>yw5027</td>
        </tr>
        <tr>
            <td>Yujie Fan</td>
            <td>yf2173</td>
        </tr>
    </table>
    </div>
    """
    , unsafe_allow_html=True
)

st.markdown('---')
st.write("### 1. Structure of the Project")

module_img = Image.open('img/module.jpg')
st.image(module_img)

st.header("All types:")
st.subheader("Plane, Car, Bird, Cat, Deer, Dog, Frog, Hourse, Ship, Truck")

# add a side bar
side_bar = st.sidebar
side_bar.title("Select your model")

# add a selectbox to the sidebar
model = side_bar.radio(
    'Model Name ',
    ('LeNet-5','VGG-16','ResNet18', 
        'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152')
)


def get_model(model):
    model_path = '../saved'
    if model == 'LeNet-5':
        net = LeNet()
        model_name = 'lenet.pth'
    elif model == 'VGG-16':
        net = Vgg16_Net()
        model_name = 'vgg16.pth'
    elif model == 'ResNet18':
        net = ResNet18()
        model_name = 'resnet18.pth'
    elif model == 'ResNet34':
        net = ResNet34()
        model_name = 'resnet34.pth'
    elif model == 'ResNet50':
        net = ResNet50()
        model_name = 'resnet50.pth'
    else:
        net = ResNet101()
        model_name = 'resnet101.pth'
    return net, os.path.join(model_path, model_name)

# upload image
upload_img = st.file_uploader("Please upload your image", type="jpg")

if upload_img is not None:
    st.image(upload_img)
    net, model_path = get_model(model)

    checkpoint = t.load(model_path, map_location=t.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    accuracy = checkpoint['acc']
    epoch = checkpoint['epoch']
    st.write("Using %s , test accuracy : %f,  epoch: %d" % (model, accuracy, epoch))

    predictor = Predictor(net, classes)
    result = predictor.predict(upload_img)
    side_bar.title("Result")
    side_bar.success("The picture is a %s" % classes[result])
