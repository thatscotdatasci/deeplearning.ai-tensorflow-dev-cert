from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import streamlit as st

from streamlit_app.abstract_classes.abstract_navigation_radio import AbstractNavigationRadio


class FashionMnistRadio(AbstractNavigationRadio):

    name = "Fashion MNIST"

    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def _action(self):
        ###
        # The Data
        ###

        with st.spinner("Loading the Fashion MNIST data..."):
            training_images, training_labels, test_images, test_labels = self._load_data()

        col1, col2 = st.beta_columns((2, 5))

        col1.markdown(f"""
        ## The Data
        
        Lets load and take a look at examples from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) data.

        **Number of training examples:** {len(training_labels)}

        **Number of validation examples:** {len(test_labels)}
        
        """)

        col1.dataframe(self._data_summary(training_labels, test_labels))

        im_num = 0
        if col1.button("Refresh example images"):
            im_num += 1
        col2.pyplot(self._get_image_selection(im_num, training_images, rows=3, cols=6))

        ###
        # The Model
        ###

        st.markdown("""
        ## The Model
        
        Below is the model defined to learn the categories of the data:
        """)

        with st.echo():
            @st.cache(show_spinner=False, allow_output_mutation=True)
            def model_fit(training_images, training_labels, epochs) -> tf.keras.Model:
                model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                ])
                model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
                model.fit(training_images, training_labels, epochs=epochs)
                return model

        with st.spinner("Training the model..."):
            model = model_fit(training_images, training_labels, epochs=5)

        st.markdown("This produces the following model summary:")
        with StringIO() as fo:
            model.summary(print_fn=lambda x: fo.write(f"{x}\n"))
            fo.seek(0)
            st.code(fo.read())

        ###
        # The Results
        ###

        st.markdown("## The Results")

        with st.spinner("Determining evaluation metrics..."):
            _, train_acc = model.evaluate(training_images, training_labels)
            _, val_acc = model.evaluate(test_images, test_labels)
        st.markdown(f"The training accuracy is {train_acc*100:.1f} %, and the validation accuracy is {val_acc*100:.1f} %.")

    @staticmethod
    @st.cache(show_spinner=False)
    def _load_data():
        mnist = tf.keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
        return training_images, training_labels, test_images, test_labels

    @staticmethod
    def _get_image_selection(_, images, rows=3, cols=3):
        n_images = rows * cols
        image_selection = np.random.choice(range(len(images)), size=n_images)

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=.01)

        for i in range(n_images):
            grid[i].imshow(images[image_selection[i]])
            grid[i].set_axis_off()

        return fig

    @st.cache(show_spinner=False)
    def _data_summary(self, training_labels: list, test_labels: list) -> pd.DataFrame:
        train_df = pd.DataFrame({"label": training_labels})
        train_df["type"] = "train"

        test_df = pd.DataFrame({"label": test_labels})
        test_df["type"] = "test"

        df = pd.merge(train_df.groupby("label").count(), test_df.groupby("label").count(), left_index=True,
                      right_index=True, suffixes=("_train", "_test"))
        df = df.reset_index()
        df["label"].replace({i: self.CLASS_NAMES[i] for i in range(10)}, inplace=True)
        df = df.rename(columns=dict(label="Category", type_train="Training Count", type_test="Test Count"))
        df = df.set_index('Category')

        return df
