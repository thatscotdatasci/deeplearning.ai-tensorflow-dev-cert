FROM python:3.8

# Create dir for the app
WORKDIR /streamlit_app/

# Install the requirements
COPY requirements.txt setup.sh ./
RUN pip install update && pip install -r requirements.txt

# Add the required code
ENV PYTHONPATH=$PYTHONPATH:streamlit_app/
COPY streamlit_app ./streamlit_app/

# Run the image as a non-root user - Heroku will not run as root
RUN useradd -m myuser
USER myuser

# Run the app - Heroku uses the CMD
# Execute setup.sh at runtime - needs to have the correct port
CMD sh setup.sh && streamlit run "./streamlit_app/main.py" --server.port "$PORT"