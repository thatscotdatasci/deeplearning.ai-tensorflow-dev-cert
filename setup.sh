#!/bin/sh
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
" > ~/.streamlit/config.toml
