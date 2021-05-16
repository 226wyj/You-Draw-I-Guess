import streamlit as st

def intro_team():
    st.markdown(
    """
    <div align="center">
    <table>
        <tr>
            <th>Name</th>
            <th>NetID</th>
            <th>Responsibility</th>
        </tr>
        <tr>
            <td>Yuejiang Wu</td>
            <td>yw5027</td>
            <td>Back End</td>
        </tr>
        <tr>
            <td>Yujie Fan</td>
            <td>yf2173</td>
            <td>Front End</td>
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

def intro_reference():
    st.markdown(
        """
        [1] http://www.cs.toronto.edu/~kriz/cifar.html

        [2] https://zhuanlan.zhihu.com/p/50468052

        [3] https://blog.csdn.net/ctwy291314/article/details/83864405

        [4] https://blog.csdn.net/daydayup_668819/article/details/79932548

        [5] https://blog.csdn.net/xinshucredit/article/details/86693872?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242

        [6] https://www.cnblogs.com/skyfsm/p/8451834.html

        [7] https://blog.csdn.net/rocling/article/details/103832980

        [8] https://www.jianshu.com/p/93990a641066

        [9] https://zhuanlan.zhihu.com/p/42706477

        [10] http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

        [11] https://arxiv.org/pdf/1409.1556.pdf

        [12] https://arxiv.org/pdf/1512.03385.pdf

        [13] https://arxiv.org/pdf/1608.06993.pdf

        [14] https://blog.csdn.net/qq_38883844/article/details/104737245

        [15] https://zhuanlan.zhihu.com/p/95680105

        [16] https://www.cnblogs.com/skyfsm/p/8451834.html

        [17] https://en.wikipedia.org/wiki/Docker_(software)

        [18] https://en.wikipedia.org/wiki/SQLite

        [19] https://desktop.arcgis.com/en/arcmap/latest/extensions/spatial-analyst/image-classification/what-is-image-classification-.htm

        [20] https://en.wikipedia.org/wiki/PyTorch

        [21] https://en.wikipedia.org/wiki/Bootstrap_(front-end_framework)

        [22] https://www.sqlite.org/index.html
        """
    )

def intro_method():
    st.markdown(
    """
    <div align="center">
    <table>
        <tr>
            <th>Module</th>
            <th>Technology</th>
        </tr>
        <tr>
            <td>Back End(Neural Network)</td>
            <td>PyTorch</td>
        </tr>
        <tr>
            <td>Front End</td>
            <td>Streamlit</td>
        </tr>
    </table>
    </div>
    """
    , unsafe_allow_html=True)


def show_img(url):
    st.markdown(
    """
    <div align="center">
        <img src={} width="70%" />
    </div>
    """.format(url)
    , unsafe_allow_html=True
    )