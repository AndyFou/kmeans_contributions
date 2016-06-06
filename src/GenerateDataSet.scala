import java.io._

import org.apache.commons.io.FileUtils

import scala.util.Random

class GenerateDataSet {
var z:Int=5

  def generateRandomFileTest(sizeOfDataSet:Int): Unit ={
    FileUtils.deleteQuietly(new File("test.txt"))
    val writer = new PrintWriter(new File("test.txt" ))

    val rg=new Random()
    writer.write(((rg.nextGaussian() * z )+20).toString + "\t" + ((rg.nextGaussian() * z )+20).toString)
    for (x <- 20 to 120 by 30) {
      for (y <- 20 to 120 by 30) {
        for(i<-1 to ((sizeOfDataSet-1)/16).toInt) {
          writer.write(("\n"+((rg.nextGaussian() * z )+x).toString + "\t" + ((rg.nextGaussian() * z )+y).toString))
        }
      }
    }

    writer.close()
  }
}
