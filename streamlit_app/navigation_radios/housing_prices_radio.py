import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import streamlit as st

from streamlit_app.abstract_classes.abstract_navigation_radio import AbstractNavigationRadio


class HousingPricesRadio(AbstractNavigationRadio):

    name = "House Price Prediction"

    def _action(self):
        ###
        # Introduction
        ###

        st.markdown("""
        ## Introduction
        
        The first exercise in the course looks at a toy example of using a perceptron to predict housing prices that 
        follow the deterministic relationship:
        """)

        st.latex("""
        \\text{House Price} = \\text{ \\textdollar 50k } + \\text{ \\textdollar 50k } \\times \\text{Number of Rooms}
        """)

        st.markdown("""
        ## TensorFlow Model
        
        The following TensorFlow code is used to train the model and make predictions. Note: The `st.cache` flag is used
        by Streamlit to cache results. 
        """)

        with st.echo():
            @st.cache
            def fit_predict(x_train, y_train, epochs, x_eval):
                model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
                model.compile(optimizer='sgd', loss='mean_squared_error', metrics=["accuracy"])
                model.fit(x_train, y_train, epochs=epochs)
                prediction = model.predict([x_eval])
                return prediction[0][0]

        st.markdown("""
        ## The Model in Action
        
        The figure below shows this very simple data. Use the slider to select how many bedrooms to use in training, 
        and what number of bedrooms to make a prediction for.
        """)


        # Create multi-columns layout
        col1, col2 = st.beta_columns((1,3))

        ###
        # User Input Column
        ###

        col1.markdown("### Training Data")

        # User specified input data
        room_lim = 10
        max_rooms = col1.slider(
            label="Max number of bedrooms to use in training data",
            min_value=1,
            max_value=room_lim,
            value=5,
            step=1
        )

        y_calc = lambda x: x * 0.5 + 0.5
        x_train = np.arange(1, max_rooms + 1)
        y_train = y_calc(x_train)

        # User specified number of epochs to run for
        col1.markdown("### Training Epochs")

        epochs = col1.slider(
            label="Training epochs",
            min_value=5,
            max_value=100,
            value=50,
            step=5
        )

        # User specified number of bedrooms to make prediction for
        col1.markdown("### Prediction")

        pred_val = col1.slider(
            label="Number of bedrooms to make a house price prediction for",
            min_value=1,
            max_value=10,
            value=7,
            step=1
        )

        # Train model and make prediction
        prediction = fit_predict(x_train, y_train, epochs, pred_val)

        col1.markdown(f"""
            #### Predicted Price: ${round(prediction*100)*1000:,.0f}
        """)

        ###
        # Figure Column
        ###

        # Input data used in training
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_train, y=y_train, name="Input Data", mode="lines+markers", line=dict(color="blue")
        ))

        # Add projection line for what the "true" values would be
        x_full = np.arange(max_rooms, room_lim + 1)
        fig.add_trace(go.Scatter(
            x=x_full, y=y_calc(x_full), mode="lines", showlegend=False, line=dict(color="blue", dash="dash")
        ))

        # Add point for predicted value
        fig.add_trace(go.Scatter(
            x=[pred_val], y=[prediction], name="Prediction", mode="markers", line=dict(color="red")
        ))
        fig.update_xaxes(title="Number of Rooms", tickmode='linear', dtick=1, range=[1, 10])
        fig.update_yaxes(title="House Prices - $k", range=[0, max(y_calc(room_lim), prediction) + 0.5])

        # Add chart to column
        col2.plotly_chart(fig, use_container_width=True)
