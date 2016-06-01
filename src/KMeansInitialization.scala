import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import scala.util.Random

class KMeansInitialization {

  def run(numClusters:Int): Unit ={
    //hide logger from console
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //create sparkContext
    val conf = new SparkConf().setAppName("HelloWorld as doulepsei")
    val sc = new SparkContext(conf)

    //load and parse the data
    val data = sc.textFile("smallData5clusters.txt")
    val parsedData = data.zipWithIndex().map{case (line,i) => i.toString + "\t" + line }.map(s => s.split('\t').map(_.toDouble)).cache()

    //find first center randomly
    val firstcenter = Random.nextInt(parsedData.count.toInt)
    var centers = new ListBuffer[Int]()
    centers += firstcenter

    //get distances of all points to first center
    val distances = parsedData.filter(x => x(0)==firstcenter).cartesian(parsedData).map(c => (c._2(0),math.sqrt(math.pow((c._1(1)-c._2(1)),2)+math.pow((c._1(2)-c._2(2)),2))))
    var distancesList = ListBuffer(distances)

    //for the rest of k: find distances, compute min distance of all points, get max distance of all min distances, set as new center
    for(x <- 1 to numClusters-1){
      val newcenter = distancesList(0).filter(_._2 != 0.0).filter(c => !centers.contains(c._1.toInt)).reduceByKey(math.min(_, _)).reduce( (a,b) => if(a._2 > b._2) a else b)
      centers += newcenter._1.toInt

      val tmp = parsedData.filter(x => x(0)==newcenter._1.toInt).cartesian(parsedData).map(c => (c._2(0),math.sqrt(math.pow((c._1(1)-c._2(1)),2)+math.pow((c._1(2)-c._2(2)),2))))
      distancesList = ListBuffer(distancesList(0).union(tmp))
    }

    centers.foreach(println)
  }
}