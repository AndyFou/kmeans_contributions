import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.{Level, Logger}

class KMeansCases(sc: SparkContext, dataFile: String, numOfCenters: Int, maxIterations:Int) {
  //hide logger from console
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val data = sc.textFile(dataFile)
  val parsedData = data.map(s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()

  def KMeansInitialCenters() = {
    val initStartTime = System.nanoTime()
    val centers = new KMeansInitialization().run(sc, dataFile, numOfCenters)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    println(s"Initialization to find centers took " + "%.3f".format(initTimeInSeconds) + " seconds.")

    val initStartTime1 = System.nanoTime()
    val model = new KMeansModel(centers)
    val clusterModel = new KMeans().setK(numOfCenters).setMaxIterations(maxIterations).setInitialModel(model).run(parsedData)
    val initTimeInSeconds1 = (System.nanoTime() - initStartTime1) / 1e9
    println(s"Initialization with custom took " + "%.3f".format(initTimeInSeconds1) + " seconds.")

    println("\nnumber of points per cluster")
    clusterModel.predict(parsedData).map(x=>(x,1)).reduceByKey((a,b)=>a+b).foreach(x=>println(x._2))

  }

  def KMeansParallel() = {
    val initStartTime = System.nanoTime()

    val clusterModel = KMeans.train(parsedData, numOfCenters, maxIterations, 1, KMeans.K_MEANS_PARALLEL)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    println(s"Initialization with KMeansParaller took " + "%.3f".format(initTimeInSeconds) + " seconds.")
    println("number of points per cluster")
    clusterModel.predict(parsedData).map(x=>(x,1)).reduceByKey((a,b)=>a+b).foreach(x=>println(x._2))
  }

  def KMeansRandom() = {
    val initStartTime = System.nanoTime()

    val clusterModel = KMeans.train(parsedData, numOfCenters, maxIterations, 1, KMeans.RANDOM)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    println(s"Initialization with KMeasRandom took " + "%.3f".format(initTimeInSeconds) + " seconds.")
    println("number of points per cluster")

    clusterModel.predict(parsedData).map(x=>(x,1)).reduceByKey((a,b)=>a+b).foreach(x=>println(x._2))
  }
}