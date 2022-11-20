# FROM ubuntu:latest
FROM python:3.8.1
COPY mlops /exp/mlops
COPY ./requirements.txt /exp/requirements.txt
COPY ./api /exp/api
COPY ./models /models
COPY ./results /results
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
ENV FLASK_APP=exp/api/api.py
WORKDIR /exp
# ENTRYPOINT ["python3", "./mlops/plot_graphs.py"]
CMD ["flask","--app", "api/app", "run", "--host", "172.17.0.2"]
# EXPOSE 5000/tcp