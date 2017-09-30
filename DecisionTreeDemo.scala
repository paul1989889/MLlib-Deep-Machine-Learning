package basic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by XieFeng on 2017/9/28.
  */
object DecisionTreeDemo {
  def main(args: Array[String]): Unit = {
    //屏蔽不需要日志记录
    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    //创建Spark配置文件
    val conf = new SparkConf().setAppName("DecisionTree").setMaster("local[4]")

    //创建SparkContext
    val sc = new SparkContext(conf)

    //设置训练数据和测试数据
    val tree1 = sc.textFile("H:/DeepLeaning_test/Tree1.txt")

    val tree2 = sc.textFile("H:/DeepLeaning_test/Tree2.txt")

    val data1 = tree1.map{
      line=>
        val parts = line.split(",")
        LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val data2 = tree2.map{
      line=>
        val parts = line.split(",")
        LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val(trainingData,testingData) = (data1,data2)


    //设置训练参数
    val numClass = 2
    val categoricalFeaturesInfo = Map[Int,Int]()
    val impurity = "gini"

    val maxDepth = 5
    val maxBins = 32

    //训练数据集
    val model = DecisionTree.trainClassifier(trainingData,numClass,categoricalFeaturesInfo,impurity,maxDepth,maxBins)


    //模型预测
    val labelAndPreds = testingData.map{
      point=>
        val prediction = model.predict(point.features)
        (point.label,prediction)
    }

    val print_predict = labelAndPreds.take(15)
    println("label"+"\t"+"prediction")
    for(i <- print_predict.indices){
      println(print_predict(i)._1+"\t"+print_predict(i)._2)
    }

    val testErr = labelAndPreds.filter(r=>r._1 != r._2).count.toDouble/testingData.count()
    println("Test Error = "+testErr)

    println("Learned classification tree model: \n"+model.toDebugString)

  }
}
