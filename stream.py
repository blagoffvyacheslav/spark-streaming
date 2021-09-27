#/spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

import json

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from datetime import datetime as dt


spark = SparkSession.builder.appName("coursework_student782_11_predict_app").getOrCreate()

checkpoint_location = "tmp/ml_checkpoint"


# upload the best model
data_path = "coursework/ml_data/*"
model_dir = "coursework/ml_data/models"
model = LogisticRegressionModel.load(model_dir + "/model_best")
output_path = "coursework/ml_data/output"


# read stream
schema = StructType.fromJson(json.loads('{"fields":[{"metadata":{},"name":"PassengerId","nullable":true,"type":"integer"},{"metadata":{},"name":"Survived","nullable":true,"type":"integer"},{"metadata":{},"name":"Pclass","nullable":true,"type":"integer"},{"metadata":{},"name":"Name","nullable":true,"type":"string"},{"metadata":{},"name":"Sex","nullable":true,"type":"string"},{"metadata":{},"name":"Age","nullable":true,"type":"integer"},{"metadata":{},"name":"SibSp","nullable":true,"type":"integer"},{"metadata":{},"name":"Parch","nullable":true,"type":"integer"},{"metadata":{},"name":"Ticket","nullable":true,"type":"string"},{"metadata":{},"name":"Fare","nullable":true,"type":"double"},{"metadata":{},"name":"Cabin","nullable":true,"type":"string"},{"metadata":{},"name":"Embarked","nullable":true,"type":"string"}],"type":"struct"}'))
features = ["Pclass,Age,SibSp,Fare,Parch,Sex,Embarked"]

data = spark \
    .readStream \
    .format("csv") \
    .schema(schema) \
    .options(header=True, maxFilesPerTrigger=1) \
    .load(data_path)


def prepare_data(df, features):
    # features
    f_columns = ",".join(features).split(",")
    # model data set
    model_data = df.select(f_columns)
    # prepare categorical columns
    text_columns = ['Sex', 'Embarked']
    output_text_columns = [c + "_index" for c in text_columns]
    for c in text_columns:
        string_indexer = StringIndexer(inputCol=c, outputCol=c + '_index').setHandleInvalid("keep")
        model_data = string_indexer.fit(model_data).transform(model_data)
    print(model_data.schema.simpleString())
    model_data.show(3)
    # update f_columns with indexed text columns
    f_columns = list(filter(lambda c: c not in text_columns, f_columns))
    f_columns += output_text_columns
    # preparation
    assembler = VectorAssembler(inputCols=f_columns, outputCol='features')
    model_data = assembler.transform(model_data)
    return model_data.select('features')


def process_batch(df, epoch):
    model_data = prepare_data(df, features)
    prediction = model.transform(model_data)
    prediction.show()
    write_to_file(prediction)


def write_to_file(df):
    load_time = dt.now().strftime("%Y%m%d%H%M%S")
    return df.write \
        .mode("append") \
        .parquet(output_path + "/p_date=" + str(load_time))


def foreach_batch_output(df):
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df\
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_location + "/" + date)\
        .start()

stream = foreach_batch_output(data)

stream.awaitTermination()