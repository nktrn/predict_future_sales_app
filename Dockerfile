FROM python:3.8


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /predict_future_sales

COPY . .
RUN mkdir model

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["run.py"]