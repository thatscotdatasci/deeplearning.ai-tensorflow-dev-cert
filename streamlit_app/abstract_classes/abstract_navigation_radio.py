from abc import ABC

import streamlit as st

from streamlit_app.abstract_classes.abstract_input_object import AbstractInputObject


class AbstractNavigationRadio(AbstractInputObject, ABC):

    _display_header = True

    def action(self):
        if self._display_header:
            st.markdown(f"# {self.name}")
        self._action()

    def _action(self):
        st.markdown("""
        ## Coming Soon!
        
        This is a placeholder page - content to come soon!
        """)
