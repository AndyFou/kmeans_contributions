import breeze.linalg.max
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ListBuffer

class KMeansExample {

  def run(): Unit ={
    //hide logger from console
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //create sparkContext
    val conf = new SparkConf().setAppName("HelloWorld as doulepsei")
    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("smallData5clusters.txt")
    val parsedData = data.map(s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()

    // Cluster the data into 5 classes using KMeans + maxNumberOfIterations
    val numClusters = 5
    val numIterations = 200
    val clusterModel = KMeans.train(parsedData, numClusters, numIterations)

    //take centers from n clusters
    val centers=clusterModel.clusterCenters
    println("prospa8o na vro ta vimata: "+clusterModel)
    centers.foreach(x=>println(x))
    //predict cluster for the dataset
    val predicted=clusterModel.predict(parsedData)

    //create id for every line in data with value 1->sizeOfData
    var ids = new ListBuffer[Int]()
    val sizeOfData=data.count().toInt
    for( counter <- 1 to sizeOfData){
      ids+=counter
    }

    //idList-->idRDD
    val idRDD=sc.parallelize(ids.toList)
    //combine all ids in pairs
    val crossIds=idRDD.cartesian(idRDD)

    //create all combinations for coordinates x,y
    val neoData=data.map(_.split('\t').map(_.toDouble))
    val crossNeoData = neoData.cartesian(neoData)

    //calculate distances for n^2 items(if 1000 lines in data->1.000.000 distances)
    val crossDistance=crossNeoData.map(c=>math.sqrt(math.pow((c._1(0)-c._2(0)),2)+math.pow((c._2(1)-c._1(1)),2)))
    //val aslkj = crossNeoData.map(x => x._1(1))

    //create all combinations for clusters for eache initial point in data
    val crossPreditedCluster=predicted.cartesian(predicted)

    //zip id,distance,predictedCluster cause its like join,but we have no key. Result: it's a mess
    val zipIdAndDistanceAndPrediction=crossIds  zip crossDistance zip crossPreditedCluster

    val tlk=zipIdAndDistanceAndPrediction.map(c=>(c._1._1._1,c._1._1._2,c._1._2,c._2._1,c._2._2))
      .persist(StorageLevel.MEMORY_AND_DISK)

    //initialize some data that we will use inside the loop
    var a:Double=0
    var bBuffer = new ListBuffer[Double]()
    var btlk:Double=0

    //Array that we'll fill with "cluster\tSilhouette" for each point in data
    var siloueteBufferForEachCluster= Array.ofDim[String](sizeOfData)


    //loop for every initial point
    for(counterForDataId<-1 to idRDD.count().toInt) {


      println("This is point: "+counterForDataId+" and belong to cluster: "
        +predicted.take(counterForDataId)(counterForDataId-1))


        //loop for every cluster
        for (counterCluster <- 0 to numClusters - 1) {
          //if cluster is same with the same with the one that is predicted then calc a
          if (predicted.take(counterForDataId)(counterForDataId-1) == counterCluster) {

            a = tlk.filter(c => c._1 == counterForDataId).filter(c => c._1 != c._2).filter(c => c._5 == counterCluster).map(c => c._3).mean
          } else {
            //else calc average for every other cluster and append it on a temp list bBuffer
            bBuffer += tlk.filter(c => c._1 == counterForDataId).filter(c => c._1 != c._2).filter(c => c._5 == counterCluster).map(c => c._3).mean()
          }

        }
          //find b
          var bList = bBuffer.toList
          btlk = bList.min
          //clear bBuffer for next loop
          bBuffer.clear()
          //calc siloute for a point and append it with its cluster on silouteList
          println(predicted.take(counterForDataId)(counterForDataId-1))
          siloueteBufferForEachCluster(counterForDataId-1) = (predicted.take(counterForDataId)(counterForDataId-1)).toString + "\t"+((btlk - a) / max(btlk, a)).toString

          println(a + " " + btlk + " " + max(a, btlk)+"  "+siloueteBufferForEachCluster(counterForDataId-1)(0)+" "+siloueteBufferForEachCluster(counterForDataId-1)(1))
    }

    val apotelesmataRDD = sc.parallelize(siloueteBufferForEachCluster.toList).map(_.split('\t').map(_.toDouble)).map(c=>(c(0).toInt,c(1)))
    println("cluster\tsiluete4eachpoint")
    apotelesmataRDD.foreach(c=>println(c._1+"\t"+c._2))
    println("Final results: Silhouette for every cluster")
    //print-calc every cluster with its average Silhouette
    for (counterCluster <- 0 to numClusters - 1) {
      println("to cluster "+counterCluster+": "+apotelesmataRDD.filter(_._1==counterCluster).map(_._2).mean())
    }
    //mean -->RDD me k lines opou k=clusters
    var neoTlkMeClusters=apotelesmataRDD.mapValues((_, 1)).reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2)).mapValues{ case (sum, count) => (1.0 * sum)/count}.collectAsMap()
    neoTlkMeClusters.map(x=>x._1.toString+" "+x._2.toString).foreach(println)

    val claster1Sum=predicted.filter(line=>line.equals(0)).count()
    val claster2Sum=predicted.filter(line=>line.equals(1)).count()
    val claster3Sum=predicted.filter(line=>line.equals(2)).count()
    val claster4Sum=predicted.filter(line=>line.equals(3)).count()
    val claster5Sum=predicted.filter(line=>line.equals(4)).count()

    println("number of claster 1 is: "+claster1Sum+"   and number 2 is:"
      +claster2Sum+"   and number 3 is:"+claster3Sum+"   and number 4 is:"
      +claster4Sum+"   and number 5 is:"+claster5Sum)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusterModel.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model

    //clusters.save(sc, "myModelPath")
    //val sameModel = KMeansModel.load(sc, "myModelPath")
    sc.stop()
  }




}
