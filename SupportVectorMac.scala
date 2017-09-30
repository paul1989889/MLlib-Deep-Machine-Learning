package basic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by XieFeng on 2017/9/29.
  */
object SupportVectorMac {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("SVM").setMaster("local[4]")

    val sc = new SparkContext(conf)

    val examples = MLUtils.loadLibSVMFile(sc,"H:/DeepLeaning_test/svm.txt").cache()

    val splits = examples.randomSplit(Array(0.6,0.4),seed = 11L)

    val training  = splits(0).cache()

    val numTraining = training.count()

    val testing = splits(1)

    val numTesting = testing.count()

    println(s"Training num : $numTraining Testing num : $numTesting")

    val numIterations = 1000

    val stepSize = 1

    val miniBatchFraction = 1.0

    val model = SVMWithSGD.train(training,numIterations,stepSize,miniBatchFraction)

    val prediction = model.predict(testing.map(_.features))

    val predictionAndLabel = prediction.zip(testing.map(_.label))

    val metrics = new MulticlassMetrics(predictionAndLabel)

    val precision = metrics.precision

    println("Precision = "+precision)
  }
}
