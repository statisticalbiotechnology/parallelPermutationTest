install.packages("devtools")
library("devtools")
install_github("bdsegal/fastPerm")

library(fastPerm)


runFastPerm <- function(nrSamples,meanY, sampleSizeList) {
  
  for (k in 1:length(sampleSizeList)) {
  
    sampleSize = sampleSizeList[[k]]
    
    e <- list(mode="vector",length=sampleSize)
    X <- list(mode="vector",length=sampleSize)
    Y <- list(mode="vector",length=sampleSize)
    Time <- list(mode="vector",length=sampleSize)
    for (i in 1:nrSamples) {
      x <- rnorm(sampleSize, mean = 5, sd = 1)
      y <- rnorm(sampleSize, mean = meanY, sd = 1)
      
      X[[i]] <- x
      Y[[i]] <- y
      
      start.time <- Sys.time()
      mStopDiffMean(x, y)
      valX = fastPerm(x, y, testStat = diffMean)
      
      end.time <- Sys.time()
      time.taken <- end.time - start.time
      
      Time[[i]] <- time.taken
      
      valX <-unlist(valX)
      
      e[[i]] <- valX[1]
      
    }
    print(e)
    write.table(X, sprintf('./dataFastPerm/data_%s/X_%s.csv',meanY, sampleSize),)
    write.table(Y, sprintf('./dataFastPerm/data_%s/y_%s.csv',meanY, sampleSize))
    write.table(format(e, scientific = FALSE), sprintf('./dataFastPerm/data_%s/error_%s.csv',meanY, sampleSize), quote = FALSE)
    write.table(format(Time, scientific = FALSE), sprintf('./dataFastPerm/data_%s/Time_%s.csv',meanY, sampleSize), quote = FALSE)
  
  }

}




runFastPerm(50, 5.6, c(5,10,50,100,150,200,250,300))
