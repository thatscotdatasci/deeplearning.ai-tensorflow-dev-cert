from typing import Type

import streamlit as st

from streamlit_app.abstract_classes.abstract_input_object import AbstractInputObject


class NavigationManager:

    _radio = None
    _radio_dict = {}

    def __init__(self, radio_classes: tuple, default_radio_class: Type[AbstractInputObject]):
        self._radio_classes = radio_classes
        self._default_radio_class = default_radio_class

        self._radio_validation()

        self._display_navigation()
        self._display_footer()
        self._display_content()

    @property
    def _radio_names(self):
        return (radio.name for radio in self._radio_classes)

    @property
    def _active_radio(self):
        return self._radio_dict[self._radio]

    def _radio_validation(self):
        for radio in self._radio_classes:
            if not issubclass(radio, AbstractInputObject):
                raise TypeError("ContentManager only accepts radios")

        if self._default_radio_class not in self._radio_classes:
            raise ValueError("The specified default radio option must be one of the passed radios")

    def _instantiate_radio(self):
        if not self._radio:
            self._radio = st.sidebar.radio(
                label="Select a page to view:",
                options=list(self._radio_names)
            )

            self._radio_dict = {
                RadioClass.name: RadioClass() for RadioClass in self._radio_classes
            }

    def _display_navigation(self):
        st.sidebar.markdown("## Navigation")
        self._instantiate_radio()

    @staticmethod
    def _display_footer():
        st.sidebar.markdown("""
        ---
        Find out more about me, and my other projects, at [ThatScotDataSci](https://thatscotdatasci.com).
        """)

    def _display_content(self):
        self._active_radio.action()
