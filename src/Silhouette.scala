import breeze.linalg.max
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}

class Silhouette(sc:SparkContext, dataFile:String, numClusters:Int, numIterations:Int) {
  def run(): Unit = {
    //hide logger from console
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val data = sc.textFile(dataFile)
    val parsedData = data.map(s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()

    val clusterModel = KMeans.train(parsedData, numClusters, numIterations)

    //take centers from n clusters
    val centers = clusterModel.clusterCenters
    //print centers from Clusters for Silhouette
    println("centers for feeding silhouette")
    centers.foreach(x => println(x))
    //predict cluster for the dataset
    val predicted = clusterModel.predict(parsedData)

    val dataWithIndex = data.zipWithIndex().
      map { case (line, i) => i.toString + "\t" + line }.map(s => s.split('\t').map(_.toDouble)).cache()// give id to points

    //print points with predicted cluster
    //(dataWithIndex zip predicted).map(x => (x._1(0), x._2)).foreach(x => println(x._1.toString + " " + x._2.toString))
    //idFrom,clusterTo,clusterFrom,distance
    val basicRdd = (dataWithIndex zip predicted).cartesian(dataWithIndex zip predicted).//cartesian (data-cluster)
      filter(c => c._1._1(0) != c._2._1(0)). //filter out points with same id
      map(c => ((c._1._1(0), c._2._2), (c._1._2, math.sqrt
      (math.pow(c._1._1(1) - c._2._1(1), 2) + math.pow(c._1._1(2) - c._2._1(2), 2)), 1)))//map idFrom,clusterTo,clusterFrom,Distance
      .reduceByKey((a, b) => (a._1, a._2 + b._2, a._3 + b._3))//find average for every possible a,b
      .mapValues { case (fromCluster, sum, count) => (fromCluster, (1.0 * sum) / count) }
      .persist(StorageLevel.MEMORY_AND_DISK)

    val onlyA = basicRdd.filter(x => x._1._2 == x._2._1).//filter a
      map(x => ((x._1._1), (x._2._1, x._2._2)))
    val onlyB = basicRdd.filter(x => x._1._2 != x._2._1). ///filter and calculate final b
      map(x => ((x._1._1), (x._1._2, x._2._1, x._2._2)))
      .reduceByKey((a, b) => (a._1, a._2, if (a._3 < b._3) a._3 else b._3)).
      map(x => ((x._1), (x._2._2, x._2._3)))

    val BothAB = onlyA.join(onlyB)// uninon a,b
    println("\n"+"silhouette coefficient")
    //BothAB.collect().foreach(x=>println("id: "+x._1.toString + "\ta:"+ x._2._1._2.toString+"\tb:"+x._2._2._2.toString))
    BothAB.map(x => (x._2._1._1, (x._2._2._2 - x._2._1._2) / max(x._2._2._2, x._2._1._2))).//calculate every si
      mapValues((_, 1))
      .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).//calculate si for every cluster
      mapValues { case (sum, count) => (1.0 * sum) / count }.
      collectAsMap().map(x => x._1.toString + " " + x._2.toString).foreach(println)

  }
}
