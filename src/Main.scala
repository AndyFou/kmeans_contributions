import org.apache.spark.{SparkConf, SparkContext}

object Main {

  def main(args: Array[String]): Unit = {

    //new GenerateDataSet().generateRandomFileTest //generate file test.txt dataset(num of points)

    val dataFile: String = "smallData5clusters.txt" //training set

    // val dataFile:String="test.txt"
    // val dataFile:String="testUEF.txt"

    val numOfCenters:Int=5
    val maxIterations:Int=200
    val conf = new SparkConf().setAppName("KMeansBigData")
    val sc = new SparkContext(conf)

    //find silhoutte
    new Silhouette(sc,dataFile,numOfCenters,maxIterations).run()

    //implementation of Initializing k-means centers
    val test=new KMeansCases(sc,dataFile,numOfCenters,maxIterations)
    println("\nStart KMeansInitialCenters")
    test.KMeansInitialCenters()
    println("\nStart KMeansParaller")
    test.KMeansParallel()
    println("\nStart KMeansRandom")
    test.KMeansRandom()

  }
}
