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

# all  classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Title
st.title("Image Classification System")
st.markdown(
    """
    
    <br><br />
    This is the final project of our team.
    <br><br />
    We implemented a simple image classification system based on deep learning method.
    <br><br />
    To read more about how the system works, click below.

    """
    , unsafe_allow_html=True
)


def presentation():
    def show_img(url):
        st.markdown(
        """
        <div align="center">
            <img src={} />
        </div>
        """.format(url)
        , unsafe_allow_html=True
        )


    # team member introduction
    st.subheader("There are two members in our team:")
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
    , unsafe_allow_html=True)

    st.markdown("## 1. Structure of the project")

    show_img("https://raw.githubusercontent.com/226wyj/You-Draw-I-Guess/main/st/img/module.jpg?token=AFSVXAXOK3C4L25CS6GBVS3AUDDOW")

    # st.markdown(
    #     """
    #     <div align="center">
    #         <img src=https://raw.githubusercontent.com/226wyj/You-Draw-I-Guess/main/st/img/module.jpg?token=AFSVXAUN5MNYWKF7MAGGPFDAUDB2M />
    #     </div>
    #     """
    #     , unsafe_allow_html=True
    # )

    st.markdown("## 2. Dataset: CIFAR-10")
    st.markdown(
        """
        The CIFAR-10 dataset consists of 60000 32x32 colour images in 10  classes, 
        with 6000 images per class. There are 50000 training images and 10000 
        test images. The dataset is divided into five training batches and one 
        test batch,  each with 10000 images. The test batch contains exactly 1000 
        randomly-selected images from each class. The training batches contain 
        the remaining images in random order, but some training batches may 
        contain more images from one class than another. Between them, the 
        training batches contain exactly 5000 images from each class. A glimpse 
        of the dataset is shown as the following figure.
        """
    )
    show_img()

more_info = st.button("More about this")

if more_info:
    presentation()



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

def classify_img():

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
