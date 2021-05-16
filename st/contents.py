import streamlit as st

def intro_team():
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


def intro_dataset():
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

def show_img(url):
    st.markdown(
    """
    <div align="center">
        <img src={} width="70%" />
    </div>
    """.format(url)
    , unsafe_allow_html=True
    )
