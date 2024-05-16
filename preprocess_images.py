from pyspark.sql.functions import udf, regexp_extract
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import cv2
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GenderDetectionPipeline") \
    .getOrCreate()

@udf(returnType=VectorUDT())
def preprocess_image(content):
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return Vectors.dense(gray.flatten().tolist())

images_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load("include/dataset/{female,male}/*.jpg")

preprocessed_df = images_df.withColumn("features", preprocess_image("content"))
preprocessed_df = preprocessed_df.withColumn("gender", regexp_extract("path", "dataset_small/([^/]+)", 1))


# Save preprocessed images to a paraquet file
preprocessed_df.write.parquet("include/Preprocessed")

spark.stop()