#R refresher, ctrl + 1 to clear console
#Run current document:	Ctrl+Alt+R
#Run from document beginning to current line:	Ctrl+Alt+B
#Run current line: enter

users <- read.csv("C:/Users/e0267429/Desktop/simplemovies.csv")

#R is index = 1, for row and col
#second col 2 to 7, take all rows, apply transpose function t()
items <- as.data.frame(t(users[,2:ncol(users)]))

#add back column names
colnames(items) <- users[,1]

#cor(items, items, use="pairwise.complete.obs")

getrecommendations <- function(target) {
  
  #compute similarity between the target user and all other users
  #c() combines its arguments to form a vector.
  #names() - Functions to get or set the names of an object
  # '%in% check whether in value is in the column name
  sims = suppressWarnings(cor(items[,target],items[,!names(items) %in% c(target)],use="pairwise.complete.obs"))
  sims = sims[1,!is.na(sims)]
  #select values in correlation >= 0
  sims <- sims[sims >= 0] 
  
  # for each item compute weighted average of all the other user ratings
  wavrats = apply(items[,names(sims)],1,function(x) weighted.mean(x, sims, na.rm=TRUE))
  wavrats = wavrats[!is.na(wavrats[])]
  
  # remove items already rated by the user
  notseenitems = row.names(items[is.na(items[,target]),])
  t = wavrats[notseenitems]  
  sort(t[!is.na(t)] , decreasing = TRUE)[1:min(5,length(t))]  # get top 5 items
}

# for simplemovies data
getrecommendations("Toby")


getrecommendations("u10")


