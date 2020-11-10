#!/bin/sh
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"user@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
" > ~/.streamlit/config.toml