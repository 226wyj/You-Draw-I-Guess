import sys
sys.path.append('..')
sys.path.append('.')
import streamlit as st
from predict import Predictor
from models.lenet import LeNet
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from models.vgg16 import Vgg16_Net
import torch as t

from st.contents import intro_team, intro_dataset, \
                intro_method, intro_reference, show_img


def main():
    # Title
    st.title("Image Classification System")
    st.markdown(
        """
        
        <br><br />
        This is the final project of our team.
        <br><br />
        We implemented a simple image classification system using deep learning method.
        <br><br />
        To read more about how the system works, click below.

        """
        , unsafe_allow_html=True
    )
    more_info = st.button("More about this")
    st.markdown("---")
    if more_info:
        presentation()
    else:
        classify_img()



def presentation():
    back_to_system = st.button("Back", key="first")
    if back_to_system:
        classify_img()

    # team member introduction
    st.markdown("## 1. Team members")
    st.markdown("There are two members in our team:")
    intro_team()
    st.markdown("---")

    # project module introduction
    st.markdown("## 2. Structure of the project")
    show_img("https://raw.githubusercontent.com/226wyj/You-Draw-I-Guess/main/st/img/module.jpg?token=AFSVXAXOK3C4L25CS6GBVS3AUDDOW")
    st.markdown("---")

    # Technology stack introduction
    st.markdown("## 3. Technology Used")
    intro_method()
    st.markdown("---")

    # dataset introduction
    st.markdown("## 4. Dataset: CIFAR-10")
    intro_dataset()
    show_img("https://raw.githubusercontent.com/226wyj/You-Draw-I-Guess/main/st/img/cifar-10.jpg?token=AFSVXAWV72PXPP7UAAQ6AITAUDDUK")
    st.markdown("---")

    # References
    st.markdown("## 5. References")
    intro_reference()
    st.markdown("---")

    back_to_system = st.button("Back", key="second")
    if back_to_system:
        classify_img()

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
    # all  classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

if __name__ == '__main__':
    main()